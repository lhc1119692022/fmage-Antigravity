#!/usr/bin/env python3
"""Generate and edit images through provider's OpenAI-compatible Image API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request
import uuid

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

# Credentials are injected by the Fmage MCP server from the external
# provider configuration. Never store a real key in this shareable plugin file.
PROVIDER_API_KEY = ""
PROVIDER_BASE_URL = ""
PROVIDER_IMAGE_MODEL = ""
DEFAULT_RESOLUTION = "2k"
CLIENT_USER_AGENT = "Fmage/0.1.0 (OpenAI-compatible image transport)"

MULTIPLE = 16
MAX_EDGE = 3840
MIN_PIXELS = 655_360
MAX_PIXELS = 8_294_400
MAX_RATIO = 3.0
MAX_RATIO_SEARCH_MARGIN = 2.98

OPTIONAL_FIELDS = {
    "quality",
    "moderation",
    "background",
    "output_format",
    "output_compression",
}

IMAGE_FIELD_AUTO = "auto"
IMAGE_FIELD_NAMES = {"image", "image[]"}
IMAGE_FIELD_RETRY_STATUSES = {400, 415, 422}


def is_image_2_model(model: str | None) -> bool:
    return (model or "").strip().lower().startswith("gpt-image-2")


def is_gpt_image_model(model: str | None) -> bool:
    return (model or "").strip().lower().startswith("gpt-image-")


def default_resolution_for_model(model: str | None) -> str:
    return "2k"


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


def resolve_image_field(requested: str, image_count: int) -> str:
    if requested in IMAGE_FIELD_NAMES:
        return requested
    if requested != IMAGE_FIELD_AUTO:
        raise ValueError(f"Unsupported multipart image field '{requested}'.")
    return "image" if image_count == 1 else "image[]"


def alternate_image_field(field_name: str) -> str:
    return "image[]" if field_name == "image" else "image"


def should_retry_image_field(error: ApiError) -> bool:
    if error.status not in IMAGE_FIELD_RETRY_STATUSES:
        return False

    body = error.body.lower()
    image_name = r"(?<![a-z])image(?:\[\]|s)?(?![a-z])"
    explicit_parameter = re.search(
        rf'["\'](?:param|parameter|field)["\']\s*:\s*["\']{image_name}["\']',
        body,
    )
    if explicit_parameter:
        return True

    patterns = (
        rf"(?:missing|required|unknown|unrecognized|unexpected|unsupported|invalid)\s+"
        rf"(?:multipart\s+)?(?:field|parameter)\s+[\"']?{image_name}",
        rf"(?:field|parameter)\s+[\"']?{image_name}[\"']?.{{0,50}}"
        r"(?:missing|required|unknown|unrecognized|unexpected|unsupported|invalid)",
        rf"(?:missing|required|not\s+provided).{{0,40}}{image_name}",
        rf"{image_name}.{{0,40}}(?:is\s+required|was\s+not\s+provided|must\s+be\s+(?:a|an)\s+(?:file|array))",
        rf"expected.{{0,40}}{image_name}.{{0,30}}(?:field|parameter|file|array)",
    )
    return any(re.search(pattern, body, re.DOTALL) for pattern in patterns)


def ceil_multiple(value: float, multiple: int = MULTIPLE) -> int:
    return max(multiple, int(math.ceil(value / multiple) * multiple))


def floor_multiple(value: float, multiple: int = MULTIPLE) -> int:
    return max(multiple, int(math.floor(value / multiple) * multiple))


def is_valid_size(width: int, height: int) -> bool:
    if width % MULTIPLE or height % MULTIPLE:
        return False
    if max(width, height) > MAX_EDGE:
        return False
    if max(width, height) / min(width, height) > MAX_RATIO:
        return False
    pixels = width * height
    return MIN_PIXELS <= pixels <= MAX_PIXELS


def parse_aspect(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip().lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[:x/]\s*(\d+(?:\.\d+)?)", value)
    if match:
        left = float(match.group(1))
        right = float(match.group(2))
        if right <= 0:
            raise ValueError(f"Invalid aspect ratio '{value}'.")
        return left / right
    numeric = float(value)
    if numeric <= 0:
        raise ValueError(f"Invalid aspect ratio '{value}'.")
    return numeric


def resolution_target_area(value: str | None) -> int:
    if not value or value.lower() == "auto":
        raise ValueError("Resolution tier is auto; no target area should be computed.")
    text = value.strip().lower()
    if text.endswith("k"):
        number = float(text[:-1])
        if number <= 0:
            raise ValueError(f"Invalid resolution '{value}'.")
        if math.isclose(number, 1.0):
            return 1024 * 1024
        if math.isclose(number, 2.0):
            return 2048 * 2048
        if math.isclose(number, 3.0):
            return min(MAX_PIXELS, 3072 * 3072)
        if math.isclose(number, 4.0):
            return MAX_PIXELS
        edge = int(number * 1024)
        return min(MAX_PIXELS, edge * edge)
    if text.endswith("px"):
        text = text[:-2]
    edge = int(text)
    return min(MAX_PIXELS, edge * edge)


def normalize_padding(width: int | float, height: int | float) -> tuple[int, int, list[str]]:
    notes: list[str] = []
    width = ceil_multiple(width)
    height = ceil_multiple(height)

    for _ in range(30):
        before = (width, height)

        if max(width, height) > MAX_EDGE:
            scale = MAX_EDGE / max(width, height)
            width = floor_multiple(width * scale)
            height = floor_multiple(height * scale)
            if "scaled_down_to_max_edge" not in notes:
                notes.append("scaled_down_to_max_edge")

        if width * height > MAX_PIXELS:
            scale = math.sqrt(MAX_PIXELS / (width * height))
            width = floor_multiple(width * scale)
            height = floor_multiple(height * scale)
            if "scaled_down_to_max_pixels" not in notes:
                notes.append("scaled_down_to_max_pixels")

        if max(width, height) / min(width, height) > MAX_RATIO:
            if width >= height:
                height = ceil_multiple(width / MAX_RATIO)
            else:
                width = ceil_multiple(height / MAX_RATIO)
            if "padded_short_edge_to_ratio" not in notes:
                notes.append("padded_short_edge_to_ratio")

        if width * height < MIN_PIXELS:
            scale = math.sqrt(MIN_PIXELS / (width * height))
            width = ceil_multiple(width * scale)
            height = ceil_multiple(height * scale)
            if "scaled_up_to_min_pixels" not in notes:
                notes.append("scaled_up_to_min_pixels")

        if before == (width, height):
            break

    if not is_valid_size(width, height):
        aspect = max(width / height, 1 / MAX_RATIO)
        aspect = min(aspect, MAX_RATIO)
        width, height = best_size_for_aspect(aspect)
        notes.append("searched_nearest_valid_size")

    return width, height, notes


def best_size_for_aspect(aspect: float, long_edge_limit: int = MAX_EDGE) -> tuple[int, int]:
    aspect = min(max(aspect, 1 / MAX_RATIO), MAX_RATIO)
    long_edge_limit = min(MAX_EDGE, floor_multiple(long_edge_limit))

    best: tuple[float, int, int, int] | None = None
    for width in range(MULTIPLE, long_edge_limit + MULTIPLE, MULTIPLE):
        for height in range(MULTIPLE, long_edge_limit + MULTIPLE, MULTIPLE):
            if not is_valid_size(width, height):
                continue
            if max(width, height) > long_edge_limit:
                continue
            ratio_error = abs(math.log((width / height) / aspect))
            area = width * height
            score = (ratio_error, -area, width, height)
            if best is None or score < best:
                best = score

    if best is None:
        return 1024, 1024
    return best[2], best[3]


def scoring_aspect(aspect: float) -> float:
    if aspect >= MAX_RATIO:
        return MAX_RATIO_SEARCH_MARGIN
    if aspect <= 1 / MAX_RATIO:
        return 1 / MAX_RATIO_SEARCH_MARGIN
    return aspect


def best_size_for_target_area(aspect: float, target_area: int) -> tuple[int, int]:
    aspect = min(max(aspect, 1 / MAX_RATIO), MAX_RATIO)
    target_area = min(max(target_area, MIN_PIXELS), MAX_PIXELS)
    target_ratio = scoring_aspect(aspect)

    best: tuple[float, float, float, int, int, int] | None = None
    for width in range(MULTIPLE, MAX_EDGE + MULTIPLE, MULTIPLE):
        for height in range(MULTIPLE, MAX_EDGE + MULTIPLE, MULTIPLE):
            if not is_valid_size(width, height):
                continue

            ratio = width / height
            if aspect > 1 and ratio < 1:
                continue
            if aspect < 1 and ratio > 1:
                continue
            if aspect >= MAX_RATIO and ratio > MAX_RATIO_SEARCH_MARGIN:
                continue
            if aspect <= 1 / MAX_RATIO and ratio < 1 / MAX_RATIO_SEARCH_MARGIN:
                continue

            area = width * height
            area_error = abs(math.log(area / target_area))
            ratio_error = abs(math.log(ratio / target_ratio))
            score = (area_error + 2 * ratio_error, ratio_error, area_error, -area, width, height)
            if best is None or score < best:
                best = score

    if best is None:
        return best_size_for_aspect(aspect)
    return best[4], best[5]


def size_from_aspect(aspect: float, resolution: str | None) -> tuple[str, list[str]]:
    notes: list[str] = []
    if not resolution or resolution.lower() == "auto":
        return "auto", ["auto_size_without_resolution_tier"]

    if aspect > MAX_RATIO:
        aspect = MAX_RATIO
        notes.append("aspect_clamped_to_3_to_1")
    elif aspect < 1 / MAX_RATIO:
        aspect = 1 / MAX_RATIO
        notes.append("aspect_clamped_to_1_to_3")

    effective_resolution = resolution
    target_area = resolution_target_area(effective_resolution)
    width, height = best_size_for_target_area(aspect, target_area)
    notes.append(f"resolution_{effective_resolution.lower()}_as_target_area")
    if width * height >= 3_686_400:
        notes.append("experimental_2k_plus_area")
    return f"{width}x{height}", notes


def infer_aspect_from_prompt(prompt: str) -> tuple[float | None, str | None]:
    text = prompt.lower()
    rules: list[tuple[float, str, tuple[str, ...]]] = [
        (9 / 16, "semantic_mobile_story", ("9:16", "story", "reel", "tiktok", "shorts", "mobile wallpaper", "phone wallpaper", "vertical poster", "phone screen")),
        (3 / 4, "semantic_portrait", ("3:4", "portrait", "headshot", "full-body", "full body", "character", "fashion", "editorial portrait", "vertical product")),
        (16 / 9, "semantic_widescreen", ("16:9", "cinematic", "widescreen", "film still", "landscape", "vehicle", "car", "suv", "environment", "panorama", "storyboard")),
        (4 / 3, "semantic_standard_landscape", ("4:3", "interior", "room", "documentary", "catalog", "standard horizontal")),
        (1.0, "semantic_square", ("1:1", "square", "icon", "avatar", "logo mark", "pattern tile", "sticker", "centered product")),
        (3.0, "semantic_ultrawide", ("3:1", "ultra-wide", "ultrawide", "banner", "header image", "wide panorama")),
    ]
    for aspect, note, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return aspect, note
    return None, None


def infer_resolution_from_quality(
    quality: str | None,
    model: str | None = None,
) -> tuple[str | None, str | None]:
    if quality == "low":
        return "1k", "resolution_inferred_from_low_quality"
    if quality == "high":
        resolution = default_resolution_for_model(model)
        return resolution, "resolution_inferred_from_high_quality"
    if quality in {"medium", "auto"} and is_image_2_model(model):
        return "2k", f"resolution_inferred_from_{quality}_quality"
    if quality in {"medium", "auto", None}:
        resolution = default_resolution_for_model(model)
        return resolution, f"resolution_{resolution}_inferred_from_model_default"
    return None, None


def resolve_size(args: argparse.Namespace, image_paths: list[Path]) -> tuple[str, list[str]]:
    if args.size:
        if args.size.lower() == "auto":
            return "auto", ["explicit_auto_size"]
        width, height = parse_size(args.size) or (0, 0)
        width, height, notes = normalize_padding(width, height)
        return f"{width}x{height}", notes

    notes: list[str] = []
    aspect = parse_aspect(args.aspect) if args.aspect else None
    if aspect is None:
        inferred_aspect, aspect_note = infer_aspect_from_prompt(read_prompt(args))
        if inferred_aspect is not None:
            aspect = inferred_aspect
            if aspect_note:
                notes.append(aspect_note)

    if aspect is None and args.command == "edit" and image_paths:
        dimensions = image_dimensions(image_paths[0])
        if dimensions:
            aspect = dimensions[0] / dimensions[1]
            if not args.resolution and not is_image_2_model(args.model):
                width, height, notes = normalize_padding(dimensions[0], dimensions[1])
                return f"{width}x{height}", ["from_first_reference_image"] + notes

    if aspect is not None:
        resolution = args.resolution
        if not resolution or resolution.lower() == "auto":
            resolution, resolution_note = infer_resolution_from_quality(args.quality, args.model)
            if resolution_note:
                notes.append(resolution_note)
        size, size_notes = size_from_aspect(aspect, resolution)
        return size, notes + size_notes

    if args.resolution and args.resolution.lower() != "auto":
        size, size_notes = size_from_aspect(1.0, args.resolution)
        return size, ["square_size_from_resolution_without_aspect"] + size_notes

    resolution, resolution_note = infer_resolution_from_quality(args.quality, args.model)
    effective_resolution = resolution or default_resolution_for_model(args.model)
    size, size_notes = size_from_aspect(1.0, effective_resolution)
    notes = [f"fallback_{effective_resolution}_square"]
    if resolution_note:
        notes.append(resolution_note)
    return size, notes + size_notes


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


def clean_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def endpoint(base_url: str, path: str) -> str:
    base = clean_base_url(base_url)
    # OpenAI-compatible providers may configure either the host root or a
    # versioned `/v1` base URL. Image endpoints always live under `/v1`.
    if path.startswith("/images/") and not re.search(r"/v1$", base, re.IGNORECASE):
        base += "/v1"
    return base + path


def pending_status(response: dict[str, Any]) -> tuple[str, str] | None:
    task_id = str(response.get("task_id") or response.get("id") or "").strip()
    status = str(response.get("status") or "").strip().lower()
    if task_id and status in {"processing", "pending", "queued", "running"}:
        return task_id, status
    return None


def task_status_endpoints(base_url: str, task_id: str, command: str) -> list[str]:
    quoted = urllib.parse.quote(task_id, safe="")
    image_path = "edits" if command == "edit" else "generations"
    paths = [
        f"/tasks/{quoted}",
        f"/task/{quoted}",
        f"/images/{image_path}/{quoted}",
    ]
    candidates = [endpoint(base_url, path) for path in paths]
    seen: set[str] = set()
    return [item for item in candidates if not (item in seen or seen.add(item))]


def json_request(
    url: str,
    body: dict[str, Any],
    api_key: str,
    timeout: int,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": CLIENT_USER_AGENT,
    }
    if extra_headers:
        headers.update(extra_headers)
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers=headers,
    )
    return perform_request(request, timeout)


def json_get(url: str, api_key: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": CLIENT_USER_AGENT,
        },
    )
    return perform_request(request, timeout)


def multipart_request(
    url: str,
    fields: dict[str, Any],
    files: list[tuple[str, Path]],
    api_key: str,
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
        "Authorization": f"Bearer {api_key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Accept": "application/json",
        "User-Agent": CLIENT_USER_AGENT,
    }
    if extra_headers:
        headers.update(extra_headers)
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers=headers,
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


def request_with_compat_retry(
    call,
    payload: dict[str, Any],
    retry_label: str,
    abort_retry: Callable[[ApiError], bool] | None = None,
    protected_fields: set[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    retries: list[str] = []
    last_error: ApiError | None = None
    try:
        return call(payload), retries
    except ApiError as error:
        last_error = error
        if abort_retry and abort_retry(error):
            raise
        if error.status not in {400, 404, 415, 422}:
            raise

    protected = protected_fields or set()
    trimmed = {
        key: value
        for key, value in payload.items()
        if key not in OPTIONAL_FIELDS or key in protected
    }
    if trimmed != payload:
        retries.append(f"{retry_label}: dropped optional fields for provider compatibility")
        try:
            return call(trimmed), retries
        except ApiError as error:
            last_error = error
            if abort_retry and abort_retry(error):
                raise
            if error.status not in {400, 404, 415, 422}:
                raise

    if trimmed.get("size") == "auto":
        without_auto_size = {key: value for key, value in trimmed.items() if key != "size"}
        retries.append(f"{retry_label}: dropped auto size for provider compatibility")
        return call(without_auto_size), retries

    if last_error:
        raise last_error
    raise RuntimeError("Provider compatibility retry failed without an API error.")


class PendingResponse(RuntimeError):
    def __init__(
        self,
        task_id: str,
        status: str,
        expired: bool,
        requested_size: str | None,
        notes: list[str],
        timing: dict[str, Any],
    ):
        super().__init__(f"remote task pending: {task_id}")
        self.result = {
            "pending": True,
            "pending_expired": expired,
            "remote_task_id": task_id,
            "remote_status": status,
            "requested_size": requested_size,
            "notes": notes,
            "warnings": [],
            "timing": timing,
        }


def poll_pending_response(
    args: argparse.Namespace,
    task_id: str,
    first_status: str,
    api_key_value: str,
    request_started: float,
    command: str,
) -> tuple[dict[str, Any] | None, list[str], str, bool]:
    total_timeout = max(0, int(getattr(args, "pending_total_timeout", 0) or 0))
    if total_timeout <= 0:
        return None, ["remote_task_pending"], first_status, False

    endpoints = task_status_endpoints(args.base_url, task_id, command)
    deadline = request_started + total_timeout
    fast_until = request_started + max(0, int(args.pending_fast_window or 0))
    fast_interval = max(1, int(args.pending_fast_interval or 20))
    slow_interval = max(1, int(args.pending_slow_interval or 45))
    selected_endpoint: str | None = None
    last_status = first_status
    notes = ["remote_task_pending"]

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, notes + ["remote_task_pending_timeout"], last_status, True

        interval = fast_interval if time.monotonic() < fast_until else slow_interval
        time.sleep(min(interval, max(0.0, remaining)))

        candidates = [selected_endpoint] if selected_endpoint else endpoints
        endpoint_errors: list[int] = []
        for candidate in [item for item in candidates if item]:
            try:
                response = json_get(candidate, api_key_value, min(args.timeout, max(1, int(deadline - time.monotonic()))))
            except ApiError as error:
                endpoint_errors.append(error.status)
                if error.status in {404, 405, 410, 429, 500, 502, 503, 504}:
                    continue
                raise
            selected_endpoint = candidate
            pending = pending_status(response)
            if pending:
                last_status = pending[1]
                break
            if isinstance(response.get("data"), list):
                return response, notes + ["remote_task_completed"], last_status, False
            status = str(response.get("status") or "").strip().lower()
            if status:
                last_status = status
            if status in {"failed", "error", "cancelled", "canceled"}:
                return None, notes + [f"remote_task_{status}"], last_status, True

        if not selected_endpoint and endpoint_errors and all(status in {404, 405, 410} for status in endpoint_errors):
            return None, notes + ["remote_task_status_endpoint_unavailable"], last_status, False


def resolve_pending_response(
    args: argparse.Namespace,
    response: dict[str, Any],
    api_key_value: str,
    request_started: float,
    timing: dict[str, Any],
    command: str,
    notes: list[str],
    requested_size: str | None,
) -> dict[str, Any]:
    pending = pending_status(response)
    if not pending:
        return response

    task_id, status = pending
    timing["pending_poll_started_at"] = iso_now()
    completed_response, pending_notes, final_status, expired = poll_pending_response(
        args,
        task_id,
        status,
        api_key_value,
        request_started,
        command,
    )
    timing["pending_poll_finished_at"] = iso_now()
    if completed_response is not None:
        notes.extend(pending_notes)
        return completed_response
    raise PendingResponse(task_id, final_status, expired, requested_size, notes + pending_notes, timing)


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
        user_agent=CLIENT_USER_AGENT,
    )


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
    timing: dict[str, Any] | None = None,
) -> Path:
    if timing is None:
        timing = {}
    timing["manifest_written_at"] = iso_now()
    manifest = {
        "command": command,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "request": request_metadata_without_prompts(request_payload),
        "requested_size": request_payload.get("size"),
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


def common_payload(args: argparse.Namespace, prompt: str, size: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": prompt,
        "size": size,
        "quality": args.quality,
        "moderation": args.moderation,
        "background": args.background,
        "output_format": args.output_format,
    }
    if args.output_compression is not None:
        payload["output_compression"] = args.output_compression
    return payload


def protected_output_fields(args: argparse.Namespace) -> set[str]:
    if args.background == "transparent":
        return {"background", "output_format"}
    return set()


def validate_common(args: argparse.Namespace) -> None:
    if args.background == "transparent" and not is_gpt_image_model(args.model):
        raise ValueError("Transparent backgrounds are supported only for GPT Image models.")
    if args.background == "transparent" and args.output_format == "jpeg":
        raise ValueError("--background transparent requires --output-format png or webp.")
    if args.output_compression is not None:
        if args.output_format not in {"jpeg", "webp"}:
            raise ValueError("--output-compression can only be used with --output-format jpeg or webp.")
        if not 0 <= args.output_compression <= 100:
            raise ValueError("--output-compression must be between 0 and 100.")

def api_key(args: argparse.Namespace) -> str:
    value = configured_value(PROVIDER_API_KEY, args.api_key_env)
    if not value:
        raise RuntimeError(
            "Missing provider API key. "
            f"{credential_help()} Environment variable {args.api_key_env} is still supported as a fallback."
        )
    return value


def run_generate(args: argparse.Namespace) -> dict[str, Any]:
    timing: dict[str, Any] = {"transport_started_at": iso_now()}
    validate_common(args)
    prompt = read_prompt(args)
    size, size_notes = resolve_size(args, [])
    payload = common_payload(args, prompt, size)

    if args.dry_run:
        return {
            "dry_run": True,
            "endpoint": endpoint(args.base_url, "/images/generations"),
            "request": payload,
            "notes": size_notes,
        }

    root = output_root(args)
    run_dir = root / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    timing["output_dir_created_at"] = iso_now()

    key = api_key(args)
    url = endpoint(args.base_url, "/images/generations")
    timing["provider_request_started_at"] = iso_now()
    request_started = time.monotonic()
    response, retry_notes = request_with_compat_retry(
        lambda body: json_request(url, body, key, args.timeout),
        payload,
        "generate",
        protected_fields=protected_output_fields(args),
    )
    timing["provider_response_completed_at"] = iso_now()
    try:
        response = resolve_pending_response(
            args,
            response,
            key,
            request_started,
            timing,
            "generate",
            size_notes + retry_notes,
            payload.get("size"),
        )
    except PendingResponse as pending:
        return pending.result
    timing["download_started_at"] = iso_now()
    images = save_response_images(response, run_dir, args.output_format, args.timeout)
    timing["download_completed_at"] = iso_now()
    image_metadata = collect_image_metadata(images)
    warnings = output_size_warnings(payload.get("size"), image_metadata)
    timing["manifest_write_started_at"] = iso_now()
    manifest_path = write_manifest(
        root,
        run_dir,
        "generate",
        payload,
        images,
        image_metadata,
        response,
        size_notes + retry_notes,
        warnings,
        timing,
    )
    return {
        "images": [str(path.resolve()) for path in images],
        "image_metadata": image_metadata,
        "requested_size": payload.get("size"),
        "manifest": str(manifest_path.resolve()),
        "notes": size_notes + retry_notes,
        "warnings": warnings,
        "timing": timing,
    }


def run_edit(args: argparse.Namespace) -> dict[str, Any]:
    timing: dict[str, Any] = {"transport_started_at": iso_now()}
    validate_common(args)
    prompt = read_prompt(args)
    root = output_root(args)

    image_paths: list[Path] = []
    if args.image:
        image_paths.extend(Path(item) for item in args.image)
    if args.use_latest:
        image_paths.extend(load_latest_images(root))
    if not image_paths:
        raise ValueError("Provide at least one --image path or use --use-latest.")
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Reference image not found: {missing}")

    size, size_notes = resolve_size(args, image_paths)
    payload = common_payload(args, prompt, size)
    image_field = resolve_image_field(args.image_field, len(image_paths))

    if args.dry_run:
        return {
            "dry_run": True,
            "endpoint": endpoint(args.base_url, "/images/edits"),
            "request": payload,
            "images": [str(path.resolve()) for path in image_paths],
            "image_field": image_field,
            "notes": size_notes,
        }

    run_dir = root / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    timing["output_dir_created_at"] = iso_now()

    key = api_key(args)
    url = endpoint(args.base_url, "/images/edits")

    def call_with_field(field_name: str):
        fields = {key_: value for key_, value in payload.items() if value is not None}
        files = [(field_name, path) for path in image_paths]
        return lambda body: multipart_request(
            url,
            {key_: value for key_, value in body.items() if value is not None},
            files,
            key,
            args.timeout,
        )

    try:
        timing["provider_request_started_at"] = iso_now()
        request_started = time.monotonic()
        response, retry_notes = request_with_compat_retry(
            call_with_field(image_field),
            payload,
            "edit",
            abort_retry=should_retry_image_field,
            protected_fields=protected_output_fields(args),
        )
    except ApiError as error:
        if not should_retry_image_field(error):
            raise
        fallback_field = alternate_image_field(image_field)
        request_started = time.monotonic()
        response, retry_notes = request_with_compat_retry(
            call_with_field(fallback_field),
            payload,
            "edit_image_field",
            abort_retry=should_retry_image_field,
            protected_fields=protected_output_fields(args),
        )
        retry_notes.append(f"edit: retried multipart field name {fallback_field} instead of {image_field}")
    timing["provider_response_completed_at"] = iso_now()
    try:
        response = resolve_pending_response(
            args,
            response,
            key,
            request_started,
            timing,
            "edit",
            size_notes + retry_notes,
            payload.get("size"),
        )
    except PendingResponse as pending:
        return pending.result

    timing["download_started_at"] = iso_now()
    images = save_response_images(response, run_dir, args.output_format, args.timeout)
    timing["download_completed_at"] = iso_now()
    image_metadata = collect_image_metadata(images)
    warnings = output_size_warnings(payload.get("size"), image_metadata)
    timing["manifest_write_started_at"] = iso_now()
    manifest_path = write_manifest(
        root,
        run_dir,
        "edit",
        payload,
        images,
        image_metadata,
        response,
        size_notes + retry_notes,
        warnings,
        timing,
    )
    return {
        "images": [str(path.resolve()) for path in images],
        "image_metadata": image_metadata,
        "requested_size": payload.get("size"),
        "manifest": str(manifest_path.resolve()),
        "notes": size_notes + retry_notes,
        "warnings": warnings,
        "timing": timing,
    }


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", help="Optimized prompt text.")
    parser.add_argument("--prompt-file", help="Path to a UTF-8 prompt file.")
    parser.add_argument("--size", help="Explicit WIDTHxHEIGHT or auto.")
    parser.add_argument("--aspect", help="Aspect ratio such as 1:1, 16:9, 3:4.")
    parser.add_argument("--resolution", help="Resolution tier such as 1k, 2k, 3k, 4k, or a long edge in px.")
    parser.add_argument("--quality", choices=["low", "medium", "high", "auto"], default="high")
    parser.add_argument("--moderation", choices=["low", "auto"], default="low")
    parser.add_argument("--background", choices=["auto", "opaque", "transparent"], default="auto")
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
    parser = argparse.ArgumentParser(description="provider GPT Image 2 generator/editor.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate images from text.")
    add_common_arguments(generate)

    edit = subparsers.add_parser("edit", help="Edit images using one or more references.")
    add_common_arguments(edit)
    edit.add_argument("--image", action="append", help="Reference image path. Repeat for multiple images.")
    edit.add_argument("--use-latest", action="store_true", help="Use latest image saved by this skill.")
    edit.add_argument(
        "--image-field",
        choices=[IMAGE_FIELD_AUTO, "image", "image[]"],
        default=IMAGE_FIELD_AUTO,
        help="Multipart field name for reference images; auto uses image for one file and image[] for multiple files.",
    )

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
