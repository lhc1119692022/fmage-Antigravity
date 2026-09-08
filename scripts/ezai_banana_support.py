#!/usr/bin/env python3
"""EzAI Nano Banana transport-only prompt, HTTP, and output helpers."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request
import uuid

from transport_common import save_response_images_from_data


PROVIDER_API_KEY = ""


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


def load_latest_images(root: Path) -> list[Path]:
    latest_path = root / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(f"No latest provider image state found at {latest_path}")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    images = [Path(item) for item in latest.get("images", [])]
    images = [path for path in images if path.exists()]
    if not images:
        raise FileNotFoundError(f"latest.json exists but contains no existing image files: {latest_path}")
    return images


def infer_aspect_from_prompt(prompt: str) -> tuple[float | None, str | None]:
    text = prompt.lower()
    rules: list[tuple[float, str, tuple[str, ...]]] = [
        (
            9 / 16,
            "semantic_mobile_story",
            (
                "9:16",
                "story",
                "reel",
                "tiktok",
                "shorts",
                "mobile wallpaper",
                "phone wallpaper",
                "vertical poster",
                "phone screen",
            ),
        ),
        (
            3 / 4,
            "semantic_portrait",
            (
                "3:4",
                "portrait",
                "headshot",
                "full-body",
                "full body",
                "character",
                "fashion",
                "editorial portrait",
                "vertical product",
            ),
        ),
        (
            16 / 9,
            "semantic_widescreen",
            (
                "16:9",
                "cinematic",
                "widescreen",
                "film still",
                "landscape",
                "vehicle",
                "car",
                "suv",
                "environment",
                "panorama",
                "storyboard",
            ),
        ),
        (
            4 / 3,
            "semantic_standard_landscape",
            ("4:3", "interior", "room", "documentary", "catalog", "standard horizontal"),
        ),
        (
            1.0,
            "semantic_square",
            ("1:1", "square", "icon", "avatar", "logo mark", "pattern tile", "sticker", "centered product"),
        ),
        (3.0, "semantic_ultrawide", ("3:1", "ultra-wide", "ultrawide", "banner", "header image", "wide panorama")),
    ]
    for aspect, note, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return aspect, note
    return None, None


def api_key(args: argparse.Namespace) -> str:
    value = configured_value(PROVIDER_API_KEY, args.api_key_env)
    if not value:
        raise RuntimeError(
            "Missing provider API key. "
            f"{credential_help()} Environment variable {args.api_key_env} is still supported as a fallback."
        )
    return value


def json_request(
    url: str,
    body: dict[str, Any],
    api_key_value: str,
    timeout: int,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key_value}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    request = urllib.request.Request(url, data=data, method="POST", headers=headers)
    return perform_request(request, timeout)


def multipart_request(
    url: str,
    fields: dict[str, Any],
    files: list[tuple[str, Path]],
    api_key_value: str,
    timeout: int,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    boundary = f"----provider-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        if value is None:
            continue
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")

    for field_name, path in files:
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{path.name}"\r\n'
                f"Content-Type: {mime_type}\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(path.read_bytes())
        chunks.append(b"\r\n")

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    data = b"".join(chunks)
    headers = {
        "Authorization": f"Bearer {api_key_value}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    if extra_headers:
        headers.update(extra_headers)
    request = urllib.request.Request(url, data=data, method="POST", headers=headers)
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


def save_response_images(
    response: dict[str, Any],
    run_dir: Path,
    output_format: str,
    timeout: int,
) -> list[Path]:
    return save_response_images_from_data(
        response,
        run_dir,
        timeout,
        preferred_format=output_format,
        base64_keys=("b64_json",),
    )
