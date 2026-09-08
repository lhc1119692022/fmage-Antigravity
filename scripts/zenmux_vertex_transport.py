#!/usr/bin/env python3
"""Generate and edit images through ZenMux Vertex AI predict endpoints."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import math
import mimetypes
import os
from pathlib import Path
import re
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
    image_dimensions,
    output_size_warnings,
    parse_size,
    request_metadata_without_prompts,
    request_prompt,
    sanitize_provider_response_metadata,
    save_response_images_from_data,
    write_latest_state,
)


DEFAULT_BASE_URL = ""
DEFAULT_MODEL = ""
DEFAULT_RESOLUTION = "2K"
DEFAULT_ASPECT = "1:1"
TRANSPORT_NAME = "zenmux-vertex"

# Credentials are injected by the Fmage MCP server from the external
# provider configuration. Never store a real key in this shareable plugin file.
PROVIDER_API_KEY = ""
PROVIDER_BASE_URL = ""
PROVIDER_IMAGE_MODEL = ""


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def credential_help() -> str:
    return "Configure the selected provider in ~/.gemini/antigravity/fmage/providers.json."


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
                f"{credential_help()}\nProvider response: {body[:1000]}"
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


def clean_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def split_model(model: str) -> tuple[str, str]:
    if "/" not in model:
        raise ValueError("ZenMux Vertex model must use provider/model format, such as google/gemini-3.1-flash-image.")
    provider, model_id = model.split("/", 1)
    provider = provider.strip()
    model_id = model_id.strip()
    if not provider or not model_id:
        raise ValueError("ZenMux Vertex model must use provider/model format.")
    return provider, model_id


def predict_endpoint(base_url: str, model: str) -> str:
    provider, model_id = split_model(model)
    root = clean_base_url(base_url)
    if not root.endswith("/v1"):
        root += "/v1"
    quoted_provider = urllib.parse.quote(provider, safe="")
    quoted_model = urllib.parse.quote(model_id, safe="")
    return f"{root}/publishers/{quoted_provider}/models/{quoted_model}:predict"


def is_openai_model(model: str) -> bool:
    return model.lower().startswith("openai/")


def output_mime_type(output_format: str | None) -> str:
    value = (output_format or "png").strip().lower()
    if value in {"jpg", "jpeg"}:
        return "image/jpeg"
    if value == "webp":
        return "image/webp"
    return "image/png"


def normalized_resolution(value: str | None) -> tuple[str, list[str]]:
    notes: list[str] = []
    if not value or value.strip().lower() == "auto":
        return DEFAULT_RESOLUTION, ["resolution_inferred_from_default_quality"]

    text = value.strip().lower()
    if text in {"512", "512px", "0.5k", ".5k"}:
        return "512", notes
    if text.endswith("px"):
        text = text[:-2]
    if text.endswith("k"):
        number = float(text[:-1])
        if math.isclose(number, 0.5):
            return "512", notes
        if math.isclose(number, 1.0):
            return "1K", notes
        if math.isclose(number, 2.0):
            return "2K", notes
        if math.isclose(number, 4.0):
            return "4K", notes
    if text in {"1024", "1"}:
        return "1K", notes
    if text in {"2048", "2"}:
        return "2K", notes
    if text in {"4096", "4"}:
        return "4K", notes
    return value.upper(), notes


def resolution_from_quality(quality: str | None) -> tuple[str, str]:
    if quality == "low":
        return "1K", "resolution_inferred_from_low_quality"
    if quality == "high":
        return "4K", "resolution_inferred_from_high_quality"
    return DEFAULT_RESOLUTION, "resolution_inferred_from_default_quality"


def resolution_from_size(width: int, height: int) -> tuple[str, str]:
    edge = max(width, height)
    if edge <= 768:
        return "512", "resolution_inferred_from_size"
    if edge <= 1408:
        return "1K", "resolution_inferred_from_size"
    if edge <= 2816:
        return "2K", "resolution_inferred_from_size"
    return "4K", "resolution_inferred_from_size"


def gcd_aspect(width: int, height: int) -> str:
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def parse_aspect_label(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[:x/]\s*(\d+(?:\.\d+)?)", text)
    if match:
        left = float(match.group(1))
        right = float(match.group(2))
        if left <= 0 or right <= 0:
            raise ValueError(f"Invalid aspect ratio '{value}'.")
        if left.is_integer() and right.is_integer():
            return gcd_aspect(int(left), int(right))
        ratio = left / right
    else:
        ratio = float(text)
        if ratio <= 0:
            raise ValueError(f"Invalid aspect ratio '{value}'.")

    candidates = {
        "1:1": 1,
        "1:4": 1 / 4,
        "1:8": 1 / 8,
        "2:3": 2 / 3,
        "3:2": 3 / 2,
        "3:4": 3 / 4,
        "4:1": 4,
        "4:3": 4 / 3,
        "4:5": 4 / 5,
        "5:4": 5 / 4,
        "8:1": 8,
        "9:16": 9 / 16,
        "16:9": 16 / 9,
        "21:9": 21 / 9,
    }
    return min(candidates, key=lambda item: abs(math.log(candidates[item] / ratio)))


def infer_aspect_from_prompt(prompt: str) -> tuple[str | None, str | None]:
    text = prompt.lower()
    rules: list[tuple[str, str, tuple[str, ...]]] = [
        ("9:16", "semantic_mobile_story", ("9:16", "story", "reel", "tiktok", "shorts", "mobile wallpaper", "phone wallpaper", "vertical poster", "phone screen")),
        ("3:4", "semantic_portrait", ("3:4", "portrait", "headshot", "full-body", "full body", "character", "fashion", "editorial portrait", "vertical product")),
        ("16:9", "semantic_widescreen", ("16:9", "cinematic", "widescreen", "film still", "landscape", "vehicle", "car", "suv", "environment", "panorama", "storyboard")),
        ("4:3", "semantic_standard_landscape", ("4:3", "interior", "room", "documentary", "catalog", "standard horizontal")),
        ("1:1", "semantic_square", ("1:1", "square", "icon", "avatar", "logo mark", "pattern tile", "sticker", "centered product")),
        ("21:9", "semantic_ultrawide", ("21:9", "ultra-wide", "ultrawide", "banner", "header image", "wide panorama")),
    ]
    for aspect, note, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return aspect, note
    return None, None


def resolve_shape(args: argparse.Namespace, references: list[str]) -> tuple[str, str, list[str]]:
    notes: list[str] = []
    if args.size and args.size.lower() != "auto":
        width, height = parse_size(args.size) or (0, 0)
        aspect = gcd_aspect(width, height)
        resolution, note = resolution_from_size(width, height)
        notes.extend(["size_mapped_to_aspect_and_resolution", note])
        return aspect, resolution, notes

    aspect = parse_aspect_label(args.aspect) if args.aspect else None
    if aspect is None:
        inferred, note = infer_aspect_from_prompt(read_prompt(args))
        if inferred:
            aspect = inferred
            if note:
                notes.append(note)

    if aspect is None and args.command == "edit" and references:
        first = references[0]
        if not is_url(first):
            dimensions = image_dimensions(Path(first))
            if dimensions:
                aspect = gcd_aspect(dimensions[0], dimensions[1])
                notes.append("aspect_from_first_reference_image")

    if aspect is None:
        aspect = DEFAULT_ASPECT
        notes.append("fallback_square_aspect")

    if args.resolution and args.resolution.lower() != "auto":
        resolution, resolution_notes = normalized_resolution(args.resolution)
        notes.extend(resolution_notes)
    else:
        resolution, note = resolution_from_quality(args.quality)
        notes.append(note)

    return aspect, resolution, notes


def validate_banana_shape(args: argparse.Namespace, aspect: str, resolution: str) -> tuple[str, str]:
    if banana_models.resolve_model(TRANSPORT_NAME, args.model, required=False) is None:
        return aspect, resolution
    explicit_dimensions = parse_size(args.size)
    if explicit_dimensions:
        aspect = banana_models.aspect_from_dimensions(TRANSPORT_NAME, args.model, *explicit_dimensions)
    elif args.aspect:
        aspect = banana_models.validate_aspect(TRANSPORT_NAME, args.model, args.aspect)
    else:
        aspect = banana_models.match_aspect_ratio(TRANSPORT_NAME, args.model, aspect)
    return (
        aspect,
        banana_models.validate_resolution(TRANSPORT_NAME, args.model, resolution),
    )


def zenmux_resolution_value(model: str, resolution: str) -> str:
    if resolution != "512px":
        return resolution
    if banana_models.resolve_model(TRANSPORT_NAME, model, required=False) is None:
        return resolution
    return "512"


def is_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def image_object(reference: str) -> dict[str, Any]:
    if is_url(reference):
        return {"gcsUri": reference}

    path = Path(reference)
    if not path.exists():
        raise FileNotFoundError(f"Reference image not found: {reference}")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"bytesBase64Encoded": encoded, "mimeType": mime_type}


def api_key(args: argparse.Namespace) -> str:
    value = configured_value(PROVIDER_API_KEY, args.api_key_env)
    if not value:
        raise RuntimeError(
            "Missing provider API key. "
            f"{credential_help()} Environment variable {args.api_key_env} is still supported as a fallback."
        )
    return value


def json_request(url: str, body: dict[str, Any], api_key_value: str, timeout: int) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key_value}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    return perform_request(request, timeout)


def perform_request(request: urllib.request.Request, timeout: int) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise ApiError(error.code, body) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Network error: {error}") from error

    try:
        return json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"API did not return JSON. First bytes: {payload[:200]!r}") from error


def build_parameters(args: argparse.Namespace, aspect: str, resolution: str) -> dict[str, Any]:
    parameters: dict[str, Any] = {"sampleCount": 1}
    mime_type = output_mime_type(args.output_format)

    if is_openai_model(args.model):
        size = args.size if args.size and args.size.lower() != "auto" else None
        if size:
            parameters["imageSize"] = size
        parameters["quality"] = args.quality
    else:
        wire_resolution = zenmux_resolution_value(args.model, resolution)
        parameters["aspectRatio"] = aspect
        parameters["sampleImageSize"] = wire_resolution

    if mime_type:
        parameters["outputOptions"] = {"mimeType": mime_type}
        if args.output_compression is not None and mime_type == "image/jpeg":
            parameters["outputOptions"]["compressionQuality"] = args.output_compression
    return parameters


def build_instance_variants(prompt: str, references: list[str]) -> list[tuple[str, dict[str, Any]]]:
    if not references:
        return [("prompt_only", {"prompt": prompt})]

    images = [image_object(item) for item in references]
    variants: list[tuple[str, dict[str, Any]]] = []
    if len(images) == 1:
        variants.append(("single_image", {"prompt": prompt, "image": images[0]}))
        variants.append(("reference_images", {"prompt": prompt, "referenceImages": [{"referenceId": 1, "referenceImage": images[0]}]}))
    else:
        reference_images = [
            {"referenceId": index, "referenceImage": image}
            for index, image in enumerate(images, start=1)
        ]
        variants.append(("reference_images", {"prompt": prompt, "referenceImages": reference_images}))
        variants.append(("first_image_plus_reference_images", {"prompt": prompt, "image": images[0], "referenceImages": reference_images}))
    return variants


def build_payload_variants(
    args: argparse.Namespace,
    prompt: str,
    references: list[str],
    aspect: str,
    resolution: str,
) -> list[tuple[str, dict[str, Any]]]:
    parameters = build_parameters(args, aspect, resolution)
    candidates: list[tuple[str, dict[str, Any]]] = []

    parameter_variants = [("standard", parameters)]
    if parameters.get("sampleImageSize") == "512":
        without_512 = dict(parameters)
        without_512.pop("sampleImageSize", None)
        parameter_variants.append(("without_sample_image_size", without_512))
    if parameters.get("outputOptions"):
        without_output = dict(parameters)
        without_output.pop("outputOptions", None)
        parameter_variants.append(("without_output_options", without_output))

    for instance_label, instance in build_instance_variants(prompt, references):
        for parameter_label, parameter_payload in parameter_variants:
            label = f"{instance_label}_{parameter_label}"
            candidates.append((label, {"instances": [instance], "parameters": parameter_payload}))

    seen: set[str] = set()
    variants: list[tuple[str, dict[str, Any]]] = []
    for label, payload in candidates:
        key = json.dumps(sanitize_payload(payload), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        variants.append((label, payload))
    return variants


def request_with_variants(
    url: str,
    variants: list[tuple[str, dict[str, Any]]],
    api_key_value: str,
    timeout: int,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    retryable_statuses = {400, 404, 415, 422}
    notes: list[str] = []
    last_error: ApiError | None = None
    for index, (label, payload) in enumerate(variants):
        try:
            response = json_request(url, payload, api_key_value, timeout)
            if index:
                notes.append(f"used_compatibility_variant_{label}")
            return response, payload, notes
        except ApiError as error:
            last_error = error
            if error.status not in retryable_statuses:
                raise
            notes.append(f"variant_{label}_failed_http_{error.status}")
    if last_error:
        raise last_error
    raise RuntimeError("No request variants were available.")


def sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"bytesBase64Encoded", "imageBytes"}:
                sanitized[key] = "[base64 image omitted]"
            else:
                sanitized[key] = sanitize_payload(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    return value


def data_response_from_predictions(response: dict[str, Any]) -> dict[str, Any]:
    predictions = response.get("predictions")
    if not isinstance(predictions, list):
        raise RuntimeError(f"API response does not contain predictions: {json.dumps(response)[:1000]}")

    data: list[dict[str, Any]] = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        image = prediction.get("image") if isinstance(prediction.get("image"), dict) else {}
        item: dict[str, Any] = {}
        encoded = (
            prediction.get("bytesBase64Encoded")
            or prediction.get("imageBytes")
            or image.get("bytesBase64Encoded")
            or image.get("imageBytes")
        )
        if encoded:
            item["b64_json"] = encoded
        mime_type = prediction.get("mimeType") or image.get("mimeType")
        if mime_type:
            item["mimeType"] = mime_type
        url = prediction.get("gcsUri") or image.get("gcsUri") or prediction.get("url")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            item["url"] = url
        if item:
            data.append(item)

    return {"data": data}


def write_manifest(
    root: Path,
    run_dir: Path,
    command: str,
    request_payload: dict[str, Any],
    images: list[Path],
    image_metadata: list[dict[str, Any]],
    response: dict[str, Any],
    notes: list[str],
    warnings: list[str],
    timing: dict[str, Any],
    requested_shape: str,
) -> Path:
    timing["manifest_written_at"] = iso_now()
    manifest = {
        "command": command,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "request": request_metadata_without_prompts(sanitize_payload(request_payload)),
        "requested_size": requested_shape,
        "images": [str(path.resolve()) for path in images],
        "image_metadata": image_metadata,
        "prompt_provenance": build_prompt_provenance(request_prompt(request_payload), response),
        "provider_response_metadata": sanitize_provider_response_metadata(response),
        "timing": timing,
        "notes": notes,
        "warnings": warnings,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_latest_state(root, manifest_path, images, manifest["created_at"])
    return manifest_path


def run_request(args: argparse.Namespace, references: list[str]) -> dict[str, Any]:
    timing: dict[str, Any] = {"transport_started_at": iso_now()}
    prompt = read_prompt(args)
    aspect, resolution, shape_notes = resolve_shape(args, references)
    aspect, resolution = validate_banana_shape(args, aspect, resolution)
    requested_shape = args.size if args.size and args.size.lower() != "auto" else f"{aspect}@{resolution}"
    variants = build_payload_variants(args, prompt, references, aspect, resolution)
    url = predict_endpoint(args.base_url, args.model)

    if args.dry_run:
        return {
            "dry_run": True,
            "endpoint": url,
            "request": sanitize_payload(variants[0][1]),
            "requested_size": requested_shape,
            "compatibility_variants": [label for label, _ in variants],
            "notes": shape_notes,
        }

    root = output_root(args)
    run_dir = root / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    timing["output_dir_created_at"] = iso_now()

    key = api_key(args)
    timing["provider_request_started_at"] = iso_now()
    response, used_payload, retry_notes = request_with_variants(url, variants, key, args.timeout)
    timing["provider_response_completed_at"] = iso_now()

    timing["download_started_at"] = iso_now()
    data_response = data_response_from_predictions(response)
    images = save_response_images_from_data(
        data_response,
        run_dir,
        args.timeout,
        preferred_format=args.output_format,
        base64_keys=("b64_json", "base64", "bytesBase64Encoded"),
        user_agent="antigravity-fmage-imagegen/1.0",
    )
    timing["download_completed_at"] = iso_now()
    image_metadata = collect_image_metadata(images)
    warnings = output_size_warnings(args.size, image_metadata)
    timing["manifest_write_started_at"] = iso_now()
    manifest_path = write_manifest(
        root,
        run_dir,
        args.command,
        used_payload,
        images,
        image_metadata,
        response,
        shape_notes + retry_notes,
        warnings,
        timing,
        requested_shape,
    )
    return {
        "images": [str(path.resolve()) for path in images],
        "image_metadata": image_metadata,
        "requested_size": requested_shape,
        "manifest": str(manifest_path.resolve()),
        "notes": shape_notes + retry_notes,
        "warnings": warnings,
        "timing": timing,
    }


def run_generate(args: argparse.Namespace) -> dict[str, Any]:
    return run_request(args, [])


def run_edit(args: argparse.Namespace) -> dict[str, Any]:
    references: list[str] = []
    if args.image:
        references.extend(args.image)
    if not references:
        raise ValueError("Provide at least one --image path/URL.")
    return run_request(args, references)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", help="Optimized prompt text.")
    parser.add_argument("--prompt-file", help="Path to a UTF-8 prompt file.")
    parser.add_argument("--size", help="Explicit WIDTHxHEIGHT or auto.")
    parser.add_argument("--aspect", help="Aspect ratio such as 1:1, 16:9, 3:4, or 21:9.")
    parser.add_argument("--resolution", help="Resolution tier such as 512, 1k, 2k, or 4k.")
    parser.add_argument("--quality", choices=["low", "medium", "high", "auto"], default="high")
    parser.add_argument("--output-format", choices=["png", "jpeg", "webp"], default="png")
    parser.add_argument("--output-compression", type=int)
    parser.add_argument("--base-url", default=configured_value(PROVIDER_BASE_URL, "PROVIDER_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=configured_value(PROVIDER_IMAGE_MODEL, "PROVIDER_IMAGE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key-env", default="PROVIDER_API_KEY")
    parser.add_argument("--output-dir")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--pending-total-timeout", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--pending-fast-window", type=int, default=120, help=argparse.SUPPRESS)
    parser.add_argument("--pending-fast-interval", type=int, default=20, help=argparse.SUPPRESS)
    parser.add_argument("--pending-slow-interval", type=int, default=45, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ZenMux Vertex AI image transport.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate images from text.")
    add_common_arguments(generate)

    edit = subparsers.add_parser("edit", help="Edit images using one or more references.")
    add_common_arguments(edit)
    edit.add_argument("--image", action="append", help="Reference image path or URL. Repeat for multiple references.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            result = run_generate(args)
        elif args.command == "edit":
            result = run_edit(args)
        else:
            parser.error("Unknown command.")
            return 2
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
