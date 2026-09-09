from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
import urllib.parse


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PLUGIN_ROOT / "scripts"
SERVER_PATH = PLUGIN_ROOT / "mcp" / "server.mjs"
sys.path.insert(0, str(SCRIPTS_DIR))

import openai_images_transport as openai
import openai_images_808_transport as transport


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WlD7xkAAAAASUVORK5CYII="
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds

    def iso_now(self) -> str:
        return f"T+{self.value:.0f}"


class PngHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(PNG_BYTES)))
        self.end_headers()
        self.wfile.write(PNG_BYTES)

    def log_message(self, format: str, *args: object) -> None:
        return


def image808_args(command: str, output_dir: Path, *extra: str) -> argparse.Namespace:
    argv = [
        command,
        "--prompt",
        "test image",
        "--base-url",
        "https://images808.example/v1",
        "--model",
        "gpt-image-2",
        "--api-key-env",
        "FMAGE_TEST_API_KEY",
        "--output-dir",
        str(output_dir),
        "--size",
        "1024x1024",
        "--timeout",
        "30",
        "--pending-total-timeout",
        "30",
        "--poll-interval",
        "1",
        *extra,
    ]
    return transport.build_parser().parse_args(argv)


class Image25RoutingTests(unittest.TestCase):
    def test_bare_image_2_5_model_is_supported(self) -> None:
        args = image808_args("generate", Path(tempfile.gettempdir()), "--model", "gpt-image-2.5")
        transport.validate_808_arguments(args)

    def test_image_2_5_accepts_xhigh_and_max(self) -> None:
        for quality in ("xhigh", "max"):
            args = image808_args(
                "generate",
                Path(tempfile.gettempdir()),
                "--model",
                "gpt-image-2.5",
                "--quality",
                quality,
            )
            transport.validate_808_arguments(args)


def call_server(config: dict[str, object], tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "providers.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        environment = {
            **os.environ,
            "FMAGE_CONFIG": str(config_path),
            "FMAGE_PYTHON": sys.executable,
        }
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        completed = subprocess.run(
            ["node", str(SERVER_PATH)],
            input=json.dumps(request) + "\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=environment,
            timeout=15,
            check=False,
        )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr or completed.stdout)
    line = completed.stdout.splitlines()[0]
    response = json.loads(line)
    if "error" in response:
        raise AssertionError(response["error"])
    return response["result"]["structuredContent"]


class EndpointAndPayloadTests(unittest.TestCase):
    def test_endpoint_merges_base_query_and_async_flag(self) -> None:
        endpoint = transport.endpoint_with_query(
            "https://api.example/v1?tenant=alpha",
            "/images/generations",
            {"async": "true"},
        )
        parsed = urllib.parse.urlsplit(endpoint)
        self.assertEqual(parsed.path, "/v1/images/generations")
        self.assertEqual(
            dict(urllib.parse.parse_qsl(parsed.query)),
            {"tenant": "alpha", "async": "true"},
        )

    def test_business_payload_reuses_openai_images_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = image808_args("generate", Path(temp_dir), "--response-format", "url")
        base_payload = openai.common_payload(args, "prompt", "1536x1024")
        image808_payload = transport.common_payload(args, "prompt", "1536x1024")
        self.assertEqual(
            {key: value for key, value in image808_payload.items() if key != "response_format"},
            base_payload,
        )
        self.assertEqual(image808_payload["response_format"], "url")

    def test_transparent_background_is_forwarded_with_png_for_async_transport(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = image808_args("generate", Path(temp_dir), "--background", "transparent")
            transport.validate_808_arguments(args)
            payload = transport.common_payload(args, "transparent prompt", "1024x1024")

        self.assertEqual(payload["background"], "transparent")
        self.assertEqual(payload["output_format"], "png")
        self.assertEqual(openai.protected_output_fields(args), {"background", "output_format"})

    def test_submission_accepts_id_and_task_id(self) -> None:
        self.assertEqual(
            transport.initial_remote_task({"id": "id-1", "status": "queued"}),
            ("id-1", "queued"),
        )
        self.assertEqual(
            transport.initial_remote_task({"task_id": "task-1", "status": "pending"}),
            ("task-1", "pending"),
        )

    def test_direct_data_response_does_not_create_remote_task(self) -> None:
        self.assertIsNone(transport.initial_remote_task({"data": [{"b64_json": "abc"}]}))

    def test_invalid_submission_response_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "neither image data nor id/task_id"):
            transport.initial_remote_task({"status": "queued"})

    def test_transport_rejects_unrelated_models_but_accepts_token_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = image808_args("generate", Path(temp_dir))
            args.model = "gemini-3-pro-image"
            with self.assertRaisesRegex(ValueError, "supports only"):
                transport.validate_808_arguments(args)
            args.model = "gpt-image-2-token"
            transport.validate_808_arguments(args)


class PollingTests(unittest.TestCase):
    def args(self, timeout: int = 30, interval: int = 1) -> argparse.Namespace:
        return argparse.Namespace(
            base_url="https://images808.example/v1",
            response_format="url",
            timeout=10,
            pending_total_timeout=timeout,
            poll_interval=interval,
        )

    def test_queued_to_in_progress_to_completed(self) -> None:
        clock = FakeClock()
        responses = iter(
            [
                {"task_id": "task-1", "status": "in_progress"},
                {"task_id": "task-1", "status": "completed", "data": [{"b64_json": "abc"}]},
            ]
        )
        requested_urls: list[str] = []

        def get(url: str, api_key: str, timeout: int) -> dict[str, object]:
            requested_urls.append(url)
            self.assertEqual(api_key, "secret")
            self.assertGreater(timeout, 0)
            return next(responses)

        timing: dict[str, object] = {}
        response, metadata, notes = transport.poll_remote_task(
            self.args(),
            "task-1",
            "queued",
            "secret",
            0.0,
            timing,
            get_fn=get,
            sleep_fn=clock.sleep,
            monotonic_fn=clock.monotonic,
            now_fn=clock.iso_now,
        )

        self.assertIn("data", response)
        self.assertEqual(
            [item["status"] for item in metadata["remote_status_history"]],
            ["queued", "in_progress", "completed"],
        )
        self.assertEqual(metadata["remote_poll_count"], 2)
        self.assertIn("remote_task_completed", notes)
        self.assertTrue(
            all(
                urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["response_format"] == ["url"]
                for url in requested_urls
            )
        )
        self.assertTrue(all("/v1/images/tasks/task-1" in url for url in requested_urls))

    def test_failed_status_keeps_remote_task_id(self) -> None:
        clock = FakeClock()
        with self.assertRaises(transport.RemoteTaskError) as raised:
            transport.poll_remote_task(
                self.args(),
                "task-failed",
                "queued",
                "secret",
                0.0,
                {},
                get_fn=lambda *_: {"status": "failed", "error": {"message": "bad input"}},
                sleep_fn=clock.sleep,
                monotonic_fn=clock.monotonic,
                now_fn=clock.iso_now,
            )
        self.assertIn("task-failed", str(raised.exception))
        self.assertIn("failed", str(raised.exception))

    def test_timeout_keeps_remote_task_id_without_resubmission(self) -> None:
        clock = FakeClock()
        get_calls = 0

        def get(*_: object) -> dict[str, object]:
            nonlocal get_calls
            get_calls += 1
            return {"status": "in_progress"}

        with self.assertRaises(transport.RemoteTaskError) as raised:
            transport.poll_remote_task(
                self.args(timeout=1, interval=5),
                "task-timeout",
                "queued",
                "secret",
                0.0,
                {},
                get_fn=get,
                sleep_fn=clock.sleep,
                monotonic_fn=clock.monotonic,
                now_fn=clock.iso_now,
            )
        self.assertEqual(get_calls, 0)
        self.assertIn("task-timeout", str(raised.exception))
        self.assertIn("timed out", str(raised.exception))


class TransportExecutionTests(unittest.TestCase):
    def test_async_base64_result_writes_remote_task_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = image808_args("generate", Path(temp_dir))
            clock = FakeClock()
            responses = iter(
                [
                    {"task_id": "task-b64", "status": "in_progress"},
                    {
                        "task_id": "task-b64",
                        "status": "completed",
                        "data": [{"b64_json": base64.b64encode(PNG_BYTES).decode("ascii")}],
                    },
                ]
            )
            with (
                mock.patch.dict(os.environ, {"FMAGE_TEST_API_KEY": "secret"}),
                mock.patch.object(
                    transport.openai,
                    "json_request",
                    return_value={"task_id": "task-b64", "status": "queued"},
                ) as submit,
                mock.patch.object(transport.openai, "json_get", side_effect=lambda *_: next(responses)),
                mock.patch.object(transport.time, "monotonic", side_effect=clock.monotonic),
                mock.patch.object(transport.time, "sleep", side_effect=clock.sleep),
            ):
                result = transport.run_generate(args)

            self.assertEqual(submit.call_count, 1)
            self.assertEqual(result["remote_task_id"], "task-b64")
            image_path = Path(result["images"][0])
            self.assertEqual(image_path.read_bytes(), PNG_BYTES)
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["remote_task_id"], "task-b64")
            self.assertEqual(
                [item["status"] for item in manifest["remote_status_history"]],
                ["queued", "in_progress", "completed"],
            )
            self.assertEqual(manifest["remote_poll_count"], 2)
            self.assertIn("pending_poll_started_at", manifest["timing"])
            self.assertIn("pending_poll_finished_at", manifest["timing"])

    def test_direct_url_result_is_downloaded_without_polling(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), PngHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        with tempfile.TemporaryDirectory() as temp_dir:
            args = image808_args("generate", Path(temp_dir))
            image_url = f"http://127.0.0.1:{server.server_port}/image.png"
            with (
                mock.patch.dict(os.environ, {"FMAGE_TEST_API_KEY": "secret"}),
                mock.patch.object(
                    transport.openai,
                    "json_request",
                    return_value={"data": [{"url": image_url}]},
                ),
                mock.patch.object(transport.openai, "json_get") as poll,
            ):
                result = transport.run_generate(args)

            poll.assert_not_called()
            self.assertNotIn("remote_task_id", result)
            self.assertEqual(Path(result["images"][0]).read_bytes(), PNG_BYTES)

    def test_edit_always_uses_image_array_for_one_or_many_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            references = [root / "one.png", root / "two.png"]
            for reference in references:
                reference.write_bytes(PNG_BYTES)

            for count in (1, 2):
                with self.subTest(count=count):
                    extra: list[str] = []
                    for reference in references[:count]:
                        extra.extend(["--image", str(reference)])
                    args = image808_args("edit", root / f"output-{count}", *extra)
                    captured_fields: list[tuple[str, Path]] = []

                    def multipart(
                        url: str,
                        fields: dict[str, object],
                        files: list[tuple[str, Path]],
                        api_key: str,
                        timeout: int,
                    ) -> dict[str, object]:
                        captured_fields.extend(files)
                        return {
                            "data": [
                                {"b64_json": base64.b64encode(PNG_BYTES).decode("ascii")}
                            ]
                        }

                    with (
                        mock.patch.dict(os.environ, {"FMAGE_TEST_API_KEY": "secret"}),
                        mock.patch.object(transport.openai, "multipart_request", side_effect=multipart),
                    ):
                        transport.run_edit(args)

                    self.assertEqual(len(captured_fields), count)
                    self.assertEqual({field for field, _ in captured_fields}, {"image[]"})


class ServerRoutingTests(unittest.TestCase):
    def config(self) -> dict[str, object]:
        return {
            "active_providers": ["image-808", "openai-images-lookalike"],
            "output_dir": "outputs",
            "cache_dir": "cache",
            "providers": {
                "image-808": {
                    "transport": "openai-images",
                    "transport_profile": "808",
                    "base_url": "https://neutral.example/v1",
                    "model": "gpt-image-2",
                    "response_format": "url",
                    "timeout": 321,
                    "api_key": "",
                },
                "openai-images-lookalike": {
                    "transport": "openai-images",
                    "base_url": "https://api.808relay.com/v1",
                    "model": "gpt-image-2",
                    "api_key": "",
                },
            },
        }

    def test_transport_profile_is_the_async_routing_switch(self) -> None:
        config = self.config()
        image808 = call_server(
            config,
            "generate_image",
            {
                "provider": "image-808",
                "prompt": "test",
                "dry_run": True,
                "verbose": True,
            },
        )
        lookalike = call_server(
            config,
            "generate_image",
            {
                "provider": "openai-images-lookalike",
                "prompt": "test",
                "dry_run": True,
                "verbose": True,
            },
        )
        self.assertEqual(image808["provider_transport"], "openai-images")
        self.assertEqual(image808["transport_profile"], "808")
        self.assertEqual(
            urllib.parse.parse_qs(urllib.parse.urlsplit(image808["endpoint"]).query)["async"],
            ["true"],
        )
        self.assertEqual(image808["request"]["response_format"], "url")
        self.assertEqual(image808["request"]["size"], "2048x2048")
        self.assertEqual(image808["request"]["quality"], "high")
        self.assertEqual(image808["remote_async"]["total_timeout_seconds"], 321)

        self.assertEqual(lookalike["provider_transport"], "openai-images")
        self.assertNotIn("async", urllib.parse.parse_qs(urllib.parse.urlsplit(lookalike["endpoint"]).query))
        self.assertNotIn("response_format", lookalike["request"])
        self.assertEqual(lookalike["request"]["size"], "2048x2048")
        self.assertEqual(lookalike["request"]["quality"], "high")

        explicit_medium = call_server(
            config,
            "generate_image",
            {
                "provider": "image-808",
                "prompt": "test",
                "quality": "medium",
                "dry_run": True,
            },
        )
        self.assertEqual(explicit_medium["request"]["size"], "2048x2048")
        self.assertEqual(explicit_medium["request"]["quality"], "medium")

    def test_legacy_transport_name_is_rejected(self) -> None:
        config = self.config()
        config["active_providers"] = ["legacy-808"]
        config["providers"] = {
            "legacy-808": {
                "transport": "808-openai-images",
                "base_url": "https://neutral.example/v1",
                "model": "gpt-image-2",
                "api_key": "",
            }
        }

        with self.assertRaisesRegex(AssertionError, "unsupported transport"):
            call_server(
                config,
                "generate_image",
                {
                    "provider": "legacy-808",
                    "prompt": "test",
                    "dry_run": True,
                },
            )

    def test_provider_status_reports_non_secret_async_configuration(self) -> None:
        status = call_server(
            self.config(),
            "get_provider_status",
            {"provider": "image-808"},
        )
        self.assertEqual(status["response_format"], "url")
        self.assertEqual(status["timeout_seconds"], 321)
        self.assertEqual(status["remote_async"]["status_path"], "/images/tasks/{task_id}")


if __name__ == "__main__":
    unittest.main()
