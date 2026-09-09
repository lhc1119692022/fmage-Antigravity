from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable
import urllib.parse

import openai_images_transport as openai


TRANSPORT_NAME = "openai-images"
TRANSPORT_PROFILE = "808"
DEFAULT_RESPONSE_FORMAT = "url"
DEFAULT_PENDING_TOTAL_TIMEOUT = 600
DEFAULT_POLL_INTERVAL = 5
SUPPORTED_MODELS = {
    "gpt-image-2",
    "gpt-image-2-token",
    "gpt-image-2.5",
    "gpt-image-2.5-flare",
    "gpt-image-2.5-sunburst",
}

PENDING_STATUSES = {"queued", "pending", "processing", "in_progress", "running"}
SUCCESS_STATUSES = {"completed", "complete", "succeeded", "success"}
FAILURE_STATUSES = {"failed", "error", "cancelled", "canceled"}
TRANSIENT_POLL_HTTP_STATUSES = {429, 500, 502, 503, 504}


class RemoteTaskError(RuntimeError):
    def __init__(self, task_id: str, status: str, detail: str):
        normalized_status = status or "unknown"
        super().__init__(
            f'808 OpenAI Images remote task "{task_id}" {detail} (last status: {normalized_status}).'
        )
        self.task_id = task_id
        self.status = normalized_status


def endpoint_with_query(base_url: str, path: str, query: dict[str, str]) -> str:
    parsed = urllib.parse.urlsplit(base_url.strip())
    base_path = parsed.path.rstrip("/")
    joined_path = f"{base_path}/{path.lstrip('/')}"
    query_keys = set(query)
    existing = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if key not in query_keys
    ]
    existing.extend((key, str(value)) for key, value in query.items())
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, joined_path, urllib.parse.urlencode(existing), parsed.fragment)
    )


def submission_endpoint(base_url: str, command: str) -> str:
    path = "/images/edits" if command == "edit" else "/images/generations"
    return endpoint_with_query(base_url, path, {"async": "true"})


def task_status_endpoint(base_url: str, task_id: str, response_format: str) -> str:
    quoted_task_id = urllib.parse.quote(task_id, safe="")
    return endpoint_with_query(
        base_url,
        f"/images/tasks/{quoted_task_id}",
        {"response_format": response_format},
    )


def task_status_endpoint_template(base_url: str, response_format: str) -> str:
    return endpoint_with_query(
        base_url,
        "/images/tasks/{task_id}",
        {"response_format": response_format},
    )


def common_payload(args: argparse.Namespace, prompt: str, size: str) -> dict[str, Any]:
    payload = openai.common_payload(args, prompt, size)
    payload["response_format"] = args.response_format
    return payload


def response_task_id(response: dict[str, Any]) -> str:
    return str(response.get("task_id") or response.get("id") or "").strip()


def response_status(response: dict[str, Any]) -> str:
    return str(response.get("status") or "").strip().lower()


def has_image_data(response: dict[str, Any]) -> bool:
    return isinstance(response.get("data"), list)


def response_error_detail(response: dict[str, Any]) -> str:
    value = response.get("error") or response.get("message")
    if value is None:
        return "failed"
    if isinstance(value, str):
        detail = value.strip()
    else:
        detail = json.dumps(value, ensure_ascii=False)
    return f"failed: {detail[:500]}" if detail else "failed"


def initial_remote_task(response: dict[str, Any]) -> tuple[str, str] | None:
    if has_image_data(response):
        return None

    task_id = response_task_id(response)
    status = response_status(response)
    if not task_id:
        raise RuntimeError(
            "808 OpenAI Images submission response contained neither image data nor id/task_id."
        )
    if status in FAILURE_STATUSES:
        raise RemoteTaskError(task_id, status, response_error_detail(response))
    if status in SUCCESS_STATUSES:
        raise RemoteTaskError(task_id, status, "completed without image data")
    if status and status not in PENDING_STATUSES:
        raise RemoteTaskError(task_id, status, "returned an unsupported task status")
    return task_id, status or "queued"


def append_status_change(
    history: list[dict[str, str]],
    status: str,
    now_fn: Callable[[], str],
) -> None:
    if history and history[-1]["status"] == status:
        return
    history.append({"status": status, "observed_at": now_fn()})


def remote_task_metadata(
    task_id: str,
    status: str,
    status_history: list[dict[str, str]],
    poll_count: int,
    poll_interval: int,
    status_endpoint: str,
) -> dict[str, Any]:
    return {
        "remote_task_id": task_id,
        "remote_status": status,
        "remote_status_history": status_history,
        "remote_poll_count": poll_count,
        "remote_poll_interval_seconds": poll_interval,
        "remote_status_endpoint": status_endpoint,
    }


def poll_remote_task(
    args: argparse.Namespace,
    task_id: str,
    first_status: str,
    api_key_value: str,
    request_started: float,
    timing: dict[str, Any],
    *,
    get_fn: Callable[[str, str, int], dict[str, Any]] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    monotonic_fn: Callable[[], float] | None = None,
    now_fn: Callable[[], str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    get_fn = get_fn or openai.json_get
    sleep_fn = sleep_fn or time.sleep
    monotonic_fn = monotonic_fn or time.monotonic
    now_fn = now_fn or openai.iso_now

    total_timeout = max(
        1,
        int(getattr(args, "pending_total_timeout", DEFAULT_PENDING_TOTAL_TIMEOUT) or 0),
    )
    poll_interval = max(1, int(getattr(args, "poll_interval", DEFAULT_POLL_INTERVAL) or 0))
    deadline = request_started + total_timeout
    status_url = task_status_endpoint(args.base_url, task_id, args.response_format)
    last_status = first_status
    status_history: list[dict[str, str]] = []
    append_status_change(status_history, last_status, now_fn)
    poll_count = 0
    notes = ["remote_task_pending"]

    timing["pending_poll_started_at"] = now_fn()
    timing["remote_task_id"] = task_id
    timing["remote_poll_interval_seconds"] = poll_interval
    timing["remote_pending_timeout_seconds"] = total_timeout

    while True:
        remaining = deadline - monotonic_fn()
        if remaining <= 0:
            timing["pending_poll_finished_at"] = now_fn()
            timing["remote_poll_count"] = poll_count
            raise RemoteTaskError(task_id, last_status, f"timed out after {total_timeout} seconds")

        sleep_fn(min(poll_interval, remaining))
        remaining = deadline - monotonic_fn()
        if remaining <= 0:
            timing["pending_poll_finished_at"] = now_fn()
            timing["remote_poll_count"] = poll_count
            raise RemoteTaskError(task_id, last_status, f"timed out after {total_timeout} seconds")

        request_timeout = min(max(1, int(args.timeout)), max(1, int(remaining)))
        try:
            response = get_fn(status_url, api_key_value, request_timeout)
        except openai.ApiError as error:
            poll_count += 1
            timing["remote_poll_count"] = poll_count
            if error.status in TRANSIENT_POLL_HTTP_STATUSES:
                note = f"remote_task_poll_http_{error.status}"
                if note not in notes:
                    notes.append(note)
                continue
            raise RemoteTaskError(task_id, last_status, f"status query failed with HTTP {error.status}") from error
        except Exception as error:
            raise RemoteTaskError(task_id, last_status, f"status query failed: {error}") from error

        poll_count += 1
        timing["remote_poll_count"] = poll_count
        status = response_status(response)

        if has_image_data(response):
            final_status = status or "completed"
            append_status_change(status_history, final_status, now_fn)
            timing["pending_poll_finished_at"] = now_fn()
            return (
                response,
                remote_task_metadata(
                    task_id,
                    final_status,
                    status_history,
                    poll_count,
                    poll_interval,
                    status_url,
                ),
                notes + ["remote_task_completed"],
            )

        if status in FAILURE_STATUSES:
            append_status_change(status_history, status, now_fn)
            timing["pending_poll_finished_at"] = now_fn()
            raise RemoteTaskError(task_id, status, response_error_detail(response))
        if status in SUCCESS_STATUSES:
            append_status_change(status_history, status, now_fn)
            timing["pending_poll_finished_at"] = now_fn()
            raise RemoteTaskError(task_id, status, "completed without image data")
        if status not in PENDING_STATUSES:
            timing["pending_poll_finished_at"] = now_fn()
            raise RemoteTaskError(
                task_id,
                status or last_status,
                "returned an invalid status response without image data",
            )

        last_status = status
        append_status_change(status_history, last_status, now_fn)


def resolve_async_response(
    args: argparse.Namespace,
    response: dict[str, Any],
    api_key_value: str,
    request_started: float,
    timing: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    remote_task = initial_remote_task(response)
    if remote_task is None:
        return response, None, []
    task_id, first_status = remote_task
    return poll_remote_task(
        args,
        task_id,
        first_status,
        api_key_value,
        request_started,
        timing,
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
    timing: dict[str, Any],
    remote_metadata: dict[str, Any] | None,
) -> Path:
    timing["manifest_written_at"] = openai.iso_now()
    manifest: dict[str, Any] = {
        "command": command,
        "transport": TRANSPORT_NAME,
        "transport_profile": TRANSPORT_PROFILE,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "request": openai.request_metadata_without_prompts(request_payload),
        "requested_size": request_payload.get("size"),
        "images": [str(path.resolve()) for path in images],
        "image_metadata": image_metadata,
        "prompt_provenance": openai.build_prompt_provenance(
            openai.request_prompt(request_payload), response
        ),
        "provider_response_metadata": openai.sanitize_provider_response_metadata(response),
        "timing": timing,
        "notes": notes,
        "warnings": warnings,
    }
    if remote_metadata:
        manifest.update(remote_metadata)

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    openai.write_latest_state(root, manifest_path, images, manifest["created_at"])
    return manifest_path


def validate_808_arguments(args: argparse.Namespace) -> None:
    openai.validate_common(args)
    if args.model not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"This transport supports only: {supported}.")
    if int(args.pending_total_timeout) <= 0:
        raise ValueError("--pending-total-timeout must be a positive integer.")
    if int(args.poll_interval) <= 0:
        raise ValueError("--poll-interval must be a positive integer.")


def dry_run_remote_async(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "enabled": True,
        "poll_interval_seconds": args.poll_interval,
        "total_timeout_seconds": args.pending_total_timeout,
        "status_endpoint": task_status_endpoint_template(args.base_url, args.response_format),
    }


def remote_result_fields(remote_metadata: dict[str, Any] | None) -> dict[str, Any]:
    return dict(remote_metadata) if remote_metadata else {}


def run_generate(args: argparse.Namespace) -> dict[str, Any]:
    timing: dict[str, Any] = {"transport_started_at": openai.iso_now()}
    validate_808_arguments(args)
    prompt = openai.read_prompt(args)
    size, size_notes = openai.resolve_size(args, [])
    payload = common_payload(args, prompt, size)
    url = submission_endpoint(args.base_url, "generate")

    if args.dry_run:
        return {
            "dry_run": True,
            "endpoint": url,
            "request": payload,
            "remote_async": dry_run_remote_async(args),
            "notes": size_notes,
        }

    root = openai.output_root(args)
    run_dir = root / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    timing["output_dir_created_at"] = openai.iso_now()

    key = openai.api_key(args)
    timing["provider_request_started_at"] = openai.iso_now()
    request_started = time.monotonic()
    response, retry_notes = openai.request_with_compat_retry(
        lambda body: openai.json_request(url, body, key, args.timeout),
        payload,
        "generate",
        protected_fields=openai.protected_output_fields(args),
    )
    timing["provider_submission_completed_at"] = openai.iso_now()
    response, remote_metadata, async_notes = resolve_async_response(
        args,
        response,
        key,
        request_started,
        timing,
    )
    timing["provider_response_completed_at"] = openai.iso_now()
    notes = size_notes + retry_notes + async_notes

    timing["download_started_at"] = openai.iso_now()
    images = openai.save_response_images(response, run_dir, args.output_format, args.timeout)
    timing["download_completed_at"] = openai.iso_now()
    image_metadata = openai.collect_image_metadata(images)
    warnings = openai.output_size_warnings(payload.get("size"), image_metadata)
    timing["manifest_write_started_at"] = openai.iso_now()
    manifest_path = write_manifest(
        root,
        run_dir,
        "generate",
        payload,
        images,
        image_metadata,
        response,
        notes,
        warnings,
        timing,
        remote_metadata,
    )
    return {
        "images": [str(path.resolve()) for path in images],
        "image_metadata": image_metadata,
        "requested_size": payload.get("size"),
        "manifest": str(manifest_path.resolve()),
        "notes": notes,
        "warnings": warnings,
        "timing": timing,
        **remote_result_fields(remote_metadata),
    }


def run_edit(args: argparse.Namespace) -> dict[str, Any]:
    timing: dict[str, Any] = {"transport_started_at": openai.iso_now()}
    validate_808_arguments(args)
    prompt = openai.read_prompt(args)
    root = openai.output_root(args)

    image_paths: list[Path] = []
    if args.image:
        image_paths.extend(Path(item) for item in args.image)
    if args.use_latest:
        image_paths.extend(openai.load_latest_images(root))
    if not image_paths:
        raise ValueError("Provide at least one --image path or use --use-latest.")
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Reference image not found: {missing}")

    size, size_notes = openai.resolve_size(args, image_paths)
    payload = common_payload(args, prompt, size)
    image_field = "image[]"
    url = submission_endpoint(args.base_url, "edit")

    if args.dry_run:
        return {
            "dry_run": True,
            "endpoint": url,
            "request": payload,
            "images": [str(path.resolve()) for path in image_paths],
            "image_field": image_field,
            "remote_async": dry_run_remote_async(args),
            "notes": size_notes,
        }

    run_dir = root / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    timing["output_dir_created_at"] = openai.iso_now()

    key = openai.api_key(args)
    files = [(image_field, path) for path in image_paths]

    def call(body: dict[str, Any]) -> dict[str, Any]:
        return openai.multipart_request(
            url,
            {key_: value for key_, value in body.items() if value is not None},
            files,
            key,
            args.timeout,
        )

    timing["provider_request_started_at"] = openai.iso_now()
    request_started = time.monotonic()
    response, retry_notes = openai.request_with_compat_retry(
        call,
        payload,
        "edit",
        abort_retry=openai.should_retry_image_field,
        protected_fields=openai.protected_output_fields(args),
    )
    timing["provider_submission_completed_at"] = openai.iso_now()
    response, remote_metadata, async_notes = resolve_async_response(
        args,
        response,
        key,
        request_started,
        timing,
    )
    timing["provider_response_completed_at"] = openai.iso_now()
    notes = size_notes + retry_notes + async_notes

    timing["download_started_at"] = openai.iso_now()
    images = openai.save_response_images(response, run_dir, args.output_format, args.timeout)
    timing["download_completed_at"] = openai.iso_now()
    image_metadata = openai.collect_image_metadata(images)
    warnings = openai.output_size_warnings(payload.get("size"), image_metadata)
    timing["manifest_write_started_at"] = openai.iso_now()
    manifest_path = write_manifest(
        root,
        run_dir,
        "edit",
        payload,
        images,
        image_metadata,
        response,
        notes,
        warnings,
        timing,
        remote_metadata,
    )
    return {
        "images": [str(path.resolve()) for path in images],
        "image_metadata": image_metadata,
        "requested_size": payload.get("size"),
        "manifest": str(manifest_path.resolve()),
        "notes": notes,
        "warnings": warnings,
        "timing": timing,
        **remote_result_fields(remote_metadata),
    }


def add_808_arguments(parser: argparse.ArgumentParser) -> None:
    openai.add_common_arguments(parser)
    parser.set_defaults(pending_total_timeout=DEFAULT_PENDING_TOTAL_TIMEOUT)
    parser.add_argument(
        "--response-format",
        choices=["url", "b64_json"],
        default=DEFAULT_RESPONSE_FORMAT,
    )
    parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="808 OpenAI Images asynchronous generator/editor."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate images from text.")
    add_808_arguments(generate)

    edit = subparsers.add_parser("edit", help="Edit local images using one or more references.")
    add_808_arguments(edit)
    edit.add_argument("--image", action="append", help="Reference image path. Repeat for multiple images.")
    edit.add_argument("--use-latest", action="store_true", help="Use latest image saved by this skill.")

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
