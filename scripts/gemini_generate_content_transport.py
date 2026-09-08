#!/usr/bin/env python3
"""Generate and edit images through the Gemini generateContent API."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import mimetypes
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

import banana_models
from transport_common import (
    build_prompt_provenance,
    collect_image_metadata,
    download_image,
    image_dimensions,
    output_size_warnings,
    parse_size,
    request_metadata_without_prompts,
    request_prompt,
    sanitize_provider_response_metadata,
    save_response_images_from_data,
    sniff_extension,
    write_latest_state,
)


TRANSPORT_NAME = "gemini-generate-content"
SUPPORTED_MODELS = frozenset(banana_models.wire_models_for(TRANSPORT_NAME))
DEFAULT_RESOLUTION = "2K"
DEFAULT_ASPECT = "1:1"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configured_value(file_value: str, env_name: str, default: str = "") -> str:
    value = (file_value or "").strip()
    if value:
        return value
    return (os.environ.get(env_name, "") or default).strip()


class ApiError(RuntimeError):
    def __init__(self, status: int, body: str):
        message = f"HTTP {status}: {body[:1000]}"
        if status in {401, 403}:
            message = (
                "provider rejected the API key or access is forbidden. "
                "Configure the selected provider in ~/.gemini/antigravity/fmage/providers.json.\n"
                f"Provider response: {body[:1000]}"
            )
        super().__init__(message)
        self.status = status
        self.body = body


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt.strip()
    raise ValueError("Provide --prompt or --prompt-file.")


def output_root(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return Path.cwd() / "outputs" / "provider-imagegen"


def normalized_model(model: str) -> str:
    value = model.strip()
    if value.startswith("models/"):
        value = value.removeprefix("models/")
    if not value or "/" in value:
        raise ValueError(
            "Gemini generateContent model must be a native model ID such as "
            "gemini-3.1-flash-image."
        )
    return value


def generate_content_endpoint(base_url: str, model: str) -> str:
    parsed = urllib.parse.urlsplit(base_url.strip())
    base_path = parsed.path.rstrip("/")
    if base_path.endswith("/v1"):
        base_path = base_path.removesuffix("/v1") + "/v1beta"
    elif not base_path.endswith("/v1beta"):
        base_path += "/v1beta"
    model_id = urllib.parse.quote(normalized_model(model), safe="")
    path = f"{base_path}/models/{model_id}:generateContent"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))


def is_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def parse_aspect(value: str) -> str:
    return banana_models.normalize_aspect(value)


def infer_aspect_from_prompt(prompt: str) -> tuple[str | None, str | None]:
    text = prompt.lower()
    rules = [
        ("9:16", "semantic_mobile_story", ("story", "reel", "tiktok", "shorts", "phone wallpaper")),
        ("3:4", "semantic_portrait", ("portrait", "headshot", "full body", "fashion")),
        ("16:9", "semantic_widescreen", ("cinematic", "widescreen", "film still", "landscape")),
        ("4:3", "semantic_standard_landscape", ("interior", "room", "documentary", "catalog")),
        ("1:1", "semantic_square", ("square", "icon", "avatar", "logo mark", "sticker")),
        ("21:9", "semantic_ultrawide", ("ultra-wide", "ultrawide", "banner", "header image")),
    ]
    for aspect, note, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return aspect, note
    return None, None


def normalized_resolution(value: str) -> str:
    text = value.strip().lower().replace(" ", "")
    aliases = {
        "512": "512px",
        "512px": "512px",
        "0.5k": "512px",
        ".5k": "512px",
        "1": "1K",
        "1k": "1K",
        "1024": "1K",
        "2": "2K",
        "2k": "2K",
        "2048": "2K",
        "4": "4K",
        "4k": "4K",
        "4096": "4K",
    }
    return aliases.get(text, value.upper())


def resolution_from_quality(quality: str | None) -> tuple[str, str]:
    if quality == "low":
        return "1K", "resolution_inferred_from_low_quality"
    if quality == "high":
        return "4K", "resolution_inferred_from_high_quality"
    return DEFAULT_RESOLUTION, "resolution_inferred_from_default_quality"


def resolve_shape(
    args: argparse.Namespace,
    references: list[str],
) -> tuple[str, str, str, list[str]]:
    notes: list[str] = []
    dimensions = parse_size(args.size)
    if dimensions:
        aspect = banana_models.aspect_from_dimensions(TRANSPORT_NAME, args.model, *dimensions)
        resolution = banana_models.resolution_from_edge(max(dimensions))
        requested_size = f"{dimensions[0]}x{dimensions[1]}"
        notes.extend(["aspect_from_explicit_size", f"explicit_size_mapped_to_{resolution.lower()}_tier"])
        return aspect, banana_models.validate_resolution(TRANSPORT_NAME, args.model, resolution), requested_size, notes

    if args.aspect:
        aspect = banana_models.validate_aspect(TRANSPORT_NAME, args.model, parse_aspect(args.aspect))
        notes.append("aspect_explicit")
    else:
        aspect = ""
        if args.command == "edit" and references:
            first = references[0]
            if not is_url(first):
                first_dimensions = image_dimensions(Path(first))
                if first_dimensions:
                    aspect = banana_models.nearest_aspect_ratio(
                        TRANSPORT_NAME,
                        args.model,
                        first_dimensions[0] / first_dimensions[1],
                    )
                    notes.append("aspect_from_first_reference_image")
        if not aspect:
            inferred, note = infer_aspect_from_prompt(read_prompt(args))
            if inferred:
                aspect = banana_models.validate_aspect(TRANSPORT_NAME, args.model, inferred)
                if note:
                    notes.append(note)
        if not aspect:
            aspect = DEFAULT_ASPECT
            notes.append("fallback_square_aspect")

    if args.resolution and args.resolution.lower() != "auto":
        resolution = normalized_resolution(args.resolution)
        notes.append(f"resolution_explicit_{resolution.lower()}")
    else:
        resolution, note = resolution_from_quality(args.quality)
        notes.append(note)
    resolution = banana_models.validate_resolution(TRANSPORT_NAME, args.model, resolution)
    return aspect, resolution, f"{resolution}@{aspect}", notes


def validate_common(args: argparse.Namespace) -> None:
    if args.model not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"Gemini generateContent accepts only these model IDs: {supported}.")
    banana_models.resolve_model(TRANSPORT_NAME, args.model)


def mime_type_from_bytes(data: bytes, fallback: str = "image/png") -> str:
    extension = sniff_extension(data, "png")
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "png": "image/png",
    }.get(extension, fallback)


def inline_image_part(reference: str, timeout: int) -> dict[str, Any]:
    if is_url(reference):
        data = download_image(reference, timeout, user_agent="fmage-gemini/1.0")
        mime_type = mime_type_from_bytes(data)
    else:
        path = Path(reference)
        if not path.is_file():
            raise FileNotFoundError(f"Reference image not found: {reference}")
        data = path.read_bytes()
        mime_type = mimetypes.guess_type(path.name)[0] or mime_type_from_bytes(data)
    return {
        "inlineData": {
            "mimeType": mime_type,
            "data": base64.b64encode(data).decode("ascii"),
        }
    }


def build_payload(
    args: argparse.Namespace,
    prompt: str,
    references: list[str],
    aspect: str,
    resolution: str,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"text": prompt}]
    parts.extend(inline_image_part(reference, args.timeout) for reference in references)
    generation_config: dict[str, Any] = {
        "responseModalities": ["TEXT", "IMAGE"],
        "imageConfig": {
            "aspectRatio": aspect,
            "imageSize": "512" if resolution == "512px" else resolution,
        },
    }
    if args.thinking_level:
        thinking_level = banana_models.resolve_thinking_level(
            TRANSPORT_NAME,
            args.model,
            args.thinking_level,
        )
        generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level.upper()}
    return {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_config,
    }


def sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[base64 image omitted]" if key == "data" and isinstance(item, str) else sanitize_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    return value


def api_key(args: argparse.Namespace) -> str:
    value = configured_value("", args.api_key_env)
    if not value:
        raise RuntimeError(
            "Missing provider API key. Configure the selected provider in "
            "~/.gemini/antigravity/fmage/providers.json."
        )
    return value


def json_request(
    url: str,
    body: dict[str, Any],
    api_key_value: str,
    timeout: int,
    auth_scheme: str = "x-goog-api-key",
) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if auth_scheme == "bearer":
        headers["Authorization"] = f"Bearer {api_key_value}"
    elif auth_scheme == "x-goog-api-key":
        headers["x-goog-api-key"] = api_key_value
    else:
        raise ValueError(f"Unsupported auth scheme: {auth_scheme}")
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        body_text = error.read().decode("utf-8", errors="replace")
        raise ApiError(error.code, body_text) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Network error: {error}") from error
    try:
        return json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"API did not return JSON. First bytes: {payload[:200]!r}") from error


def data_response_from_candidates(response: dict[str, Any]) -> dict[str, Any]:
    candidates = response.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError(f"API response does not contain candidates: {json.dumps(response)[:1000]}")
    data: list[dict[str, Any]] = []
    for candidate in candidates:
        content = candidate.get("content") if isinstance(candidate, dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if not isinstance(inline, dict) or not inline.get("data"):
                continue
            data.append(
                {
                    "b64_json": inline["data"],
                    "mimeType": inline.get("mimeType") or inline.get("mime_type"),
                }
            )
    if not data:
        raise RuntimeError("Gemini response contained no inline image data.")
    return {"data": data}


def write_manifest(
    root: Path,
    run_dir: Path,
    command: str,
    payload: dict[str, Any],
    requested_size: str,
    images: list[Path],
    image_metadata: list[dict[str, Any]],
    response: dict[str, Any],
    notes: list[str],
    warnings: list[str],
    timing: dict[str, Any],
) -> Path:
    timing["manifest_written_at"] = iso_now()
    manifest = {
        "command": command,
        "transport": TRANSPORT_NAME,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "request": request_metadata_without_prompts(sanitize_payload(payload)),
        "requested_size": requested_size,
        "images": [str(path.resolve()) for path in images],
        "image_metadata": image_metadata,
        "prompt_provenance": build_prompt_provenance(request_prompt(payload), response),
        "provider_response_metadata": sanitize_provider_response_metadata(response),
        "timing": timing,
        "notes": notes,
        "warnings": warnings,
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_latest_state(root, path, images, manifest["created_at"])
    return path


def run_request(args: argparse.Namespace, references: list[str]) -> dict[str, Any]:
    timing: dict[str, Any] = {"transport_started_at": iso_now()}
    validate_common(args)
    prompt = read_prompt(args)
    aspect, resolution, requested_size, notes = resolve_shape(args, references)
    payload = build_payload(args, prompt, references, aspect, resolution)
    url = generate_content_endpoint(args.base_url, args.model)
    if args.dry_run:
        return {
            "dry_run": True,
            "endpoint": url,
            "request": sanitize_payload(payload),
            "requested_size": requested_size,
            "requested_resolution": resolution,
            "requested_aspect_ratio": aspect,
            "notes": notes,
        }

    root = output_root(args)
    run_dir = root / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    timing["output_dir_created_at"] = iso_now()
    timing["provider_request_started_at"] = iso_now()
    response = json_request(url, payload, api_key(args), args.timeout, args.auth_scheme)
    timing["provider_response_completed_at"] = iso_now()
    timing["download_started_at"] = iso_now()
    images = save_response_images_from_data(
        data_response_from_candidates(response),
        run_dir,
        args.timeout,
        preferred_format=args.output_format,
        base64_keys=("b64_json",),
    )
    timing["download_completed_at"] = iso_now()
    image_metadata = collect_image_metadata(images)
    warnings = output_size_warnings(args.size, image_metadata)
    timing["manifest_write_started_at"] = iso_now()
    manifest = write_manifest(
        root,
        run_dir,
        args.command,
        payload,
        requested_size,
        images,
        image_metadata,
        response,
        notes,
        warnings,
        timing,
    )
    return {
        "images": [str(path.resolve()) for path in images],
        "image_metadata": image_metadata,
        "requested_size": requested_size,
        "requested_resolution": resolution,
        "requested_aspect_ratio": aspect,
        "manifest": str(manifest.resolve()),
        "notes": notes,
        "warnings": warnings,
        "timing": timing,
    }


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--size")
    parser.add_argument("--aspect")
    parser.add_argument("--resolution")
    parser.add_argument("--quality", choices=["low", "medium", "high", "auto"], default="high")
    parser.add_argument("--thinking-level", choices=["minimal", "high"])
    parser.add_argument("--output-format", choices=["png", "jpeg", "webp"], default="png")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="PROVIDER_API_KEY")
    parser.add_argument("--auth-scheme", choices=["x-goog-api-key", "bearer"], default="x-goog-api-key")
    parser.add_argument("--output-dir")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--pending-total-timeout", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--pending-fast-window", type=int, default=120, help=argparse.SUPPRESS)
    parser.add_argument("--pending-fast-interval", type=int, default=20, help=argparse.SUPPRESS)
    parser.add_argument("--pending-slow-interval", type=int, default=45, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemini generateContent image transport.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    add_common_arguments(generate)
    edit = subparsers.add_parser("edit")
    add_common_arguments(edit)
    edit.add_argument("--image", action="append")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        references = list(args.image or []) if args.command == "edit" else []
        if args.command == "edit" and not references:
            raise ValueError("Provide at least one --image path/URL.")
        result = run_request(args, references)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
