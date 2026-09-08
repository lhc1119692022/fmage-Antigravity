from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = PLUGIN_ROOT / "mcp" / "server.mjs"
STANDARD_IMAGE_TOOLS = {
    "generate_image",
    "generate_image_batch",
    "edit_image",
    "edit_image_batch",
}
BASELINE_TOOL_NAMES = STANDARD_IMAGE_TOOLS | {
    "regress_image",
    "trace_image_job_plan",
    "probe_image_generation",
    "get_image_task_status",
    "get_provider_status",
}
BASELINE_INSTRUCTIONS = (
    "Use Fmage image tools. For each image request, make one concise understanding-and-expansion pass "
    "in the same turn, then call the matching base image tool immediately. Keep simple requests short; "
    "expand only missing visual constraints. Do not create a separate plan, prompt-optimizer pass, "
    "trace/status preflight, or repeated rewrite. For edits, identify reference-image roles and preserve "
    "required text, layout, and other locked details."
)
TRANSPARENT_TEST_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WlD7xkAAAAASUVORK5CYII="
)


def provider(model: str = "gpt-image-2", transport: str = "openai-images") -> dict[str, object]:
    return {
        "transport": transport,
        "base_url": "https://example.invalid/v1",
        "model": model,
        "api_key": "",
    }


def provider_config(
    active_providers: list[str],
    providers: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "active_providers": active_providers,
        "output_dir": "outputs",
        "cache_dir": "cache",
        "providers": providers,
    }


class ServerSession:
    def __init__(self, config: dict[str, object]) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        config_path = Path(self.temporary_directory.name) / "providers.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        environment = {
            **os.environ,
            "FMAGE_CONFIG": str(config_path),
            "FMAGE_PYTHON": sys.executable,
        }
        self.process = subprocess.Popen(
            ["node", str(SERVER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=environment,
        )
        self.next_id = 1

    def call(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if self.process.stdin is None or self.process.stdout is None:
            raise AssertionError("Fmage test server pipes are unavailable")
        request_id = self.next_id
        self.next_id += 1
        message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr is not None else ""
            raise AssertionError(stderr or "Fmage test server exited without a response")
        response = json.loads(line)
        if response.get("id") != request_id:
            raise AssertionError(f"Expected response id {request_id}, got: {response!r}")
        return response

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stderr is not None:
            self.process.stderr.close()
        self.temporary_directory.cleanup()


_SERVER_SESSIONS: dict[str, ServerSession] = {}


def call_server(
    config: dict[str, object],
    method: str,
    params: dict[str, object],
) -> dict[str, object]:
    key = json.dumps(config, sort_keys=True)
    session = _SERVER_SESSIONS.get(key)
    if session is None:
        session = ServerSession(config)
        _SERVER_SESSIONS[key] = session
    return session.call(method, params)


def tearDownModule() -> None:
    for session in _SERVER_SESSIONS.values():
        session.close()
    _SERVER_SESSIONS.clear()


def tool_map(response: dict[str, object]) -> dict[str, dict[str, object]]:
    tools = response["result"]["tools"]
    return {tool["name"]: tool for tool in tools}


def all_property_names(value: object) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            names.update(properties)
        for child in value.values():
            names.update(all_property_names(child))
    elif isinstance(value, list):
        for child in value:
            names.update(all_property_names(child))
    return names


class PromptPolicyIsolationTests(unittest.TestCase):
    def clean_config(self) -> dict[str, object]:
        return provider_config(["image-2"], {"image-2": provider()})

    def call_image_tool(
        self,
        config: dict[str, object],
        name: str = "generate_image",
        **arguments: object,
    ) -> dict[str, object]:
        return call_server(
            config,
            "tools/call",
            {
                "name": name,
                "arguments": {"dry_run": True, "verbose": True, **arguments},
            },
        )

    def test_tools_and_initialize_use_one_pass_fast_path(self) -> None:
        config = self.clean_config()
        response = call_server(config, "tools/list", {})
        tools = tool_map(response)

        self.assertEqual(set(tools), BASELINE_TOOL_NAMES)
        serialized = json.dumps(tools, ensure_ascii=False).lower()
        for removed_detail in (
            "dall-e",
            "prepare_prompt",
            "prompt_profile",
            "prompt_policy",
            "provider_prompt",
            "prompt_check_id",
            "prompt_session_id",
        ):
            self.assertNotIn(removed_detail, serialized)

        for name in STANDARD_IMAGE_TOOLS:
            schema = tools[name]["inputSchema"]
            self.assertIn("prompt", all_property_names(schema))
            self.assertNotIn("provider_prompt", all_property_names(schema))
            self.assertNotIn("prompt_check_id", all_property_names(schema))
            self.assertIn("resolution_user_requested", all_property_names(schema))
            self.assertIn("output_format_user_requested", all_property_names(schema))
            self.assertIn("thinking_level_user_requested", all_property_names(schema))

        prompt_description = tools["generate_image"]["inputSchema"]["properties"]["prompt"]["description"]
        self.assertIn("one concise understanding-and-expansion pass", prompt_description)
        self.assertIn("Do not include analysis or a planning preamble", prompt_description)

        initialize = call_server(
            config,
            "initialize",
            {"protocolVersion": "2025-11-25"},
        )
        self.assertEqual(initialize["result"]["instructions"], BASELINE_INSTRUCTIONS)
        self.assertIn("call the matching base image tool immediately", BASELINE_INSTRUCTIONS)
        self.assertNotIn("prepare_prompt", initialize["result"]["instructions"])
        self.assertNotIn("DALL-E", initialize["result"]["instructions"])

    def test_direct_prompt_is_submitted_without_server_prompt_preparation(self) -> None:
        config = self.clean_config()
        prompt = "A finished editorial product photograph with controlled studio lighting."
        response = self.call_image_tool(config, prompt=prompt)
        self.assertNotIn("error", response)
        result = response["result"]["structuredContent"]
        self.assertEqual(result["request"]["prompt"], prompt)
        self.assertEqual(result["revised_prompt_submitted"], prompt)
        self.assertNotIn("prompt_preparation", result)
        self.assertNotIn("prompt_policy", result)
        self.assertNotIn("prompt_profile", result)
        self.assertNotIn("provider_prompt", result)

    def test_default_quality_high_keeps_resolution_at_2k(self) -> None:
        config = self.clean_config()
        response = self.call_image_tool(config, prompt="plain test image")
        self.assertNotIn("error", response)
        request = response["result"]["structuredContent"]["request"]
        self.assertEqual(request["quality"], "high")
        self.assertEqual(request["size"], "2048x2048")

    def test_high_quality_prompt_semantic_uses_4k(self) -> None:
        config = self.clean_config()
        response = self.call_image_tool(config, prompt="A high-quality product image.")
        self.assertNotIn("error", response)
        request = response["result"]["structuredContent"]["request"]
        self.assertEqual(request["quality"], "high")
        self.assertEqual(request["size"], "2880x2880")

    def test_visual_8k_language_does_not_become_delivery_resolution(self) -> None:
        config = self.clean_config()
        prompt = "一张产品摄影图，具备8K超高分辨率、极致清晰细节和真实材质。"
        response = self.call_image_tool(
            config,
            prompt=prompt,
            resolution="4k",
            size="4096x4096",
            resolution_user_requested=True,
        )
        self.assertNotIn("error", response)
        result = response["result"]["structuredContent"]
        self.assertEqual(result["request"]["prompt"], prompt)
        self.assertNotIn("resolution", result["request"])
        self.assertEqual(result["request"]["size"], "2048x2048")
        self.assertTrue(any("visual-quality language" in warning for warning in result["warnings"]))

    def test_unrequested_jpeg_falls_back_to_png(self) -> None:
        config = self.clean_config()
        response = self.call_image_tool(config, prompt="plain test image", output_format="jpeg")
        self.assertNotIn("error", response)
        result = response["result"]["structuredContent"]
        self.assertEqual(result["request"]["output_format"], "png")
        self.assertTrue(any("output_format_user_requested" in warning for warning in result["warnings"]))

    def test_explicit_jpeg_remains_a_parameter(self) -> None:
        config = self.clean_config()
        response = self.call_image_tool(
            config,
            prompt="plain test image",
            output_format="jpeg",
            output_format_user_requested=True,
        )
        self.assertNotIn("error", response)
        result = response["result"]["structuredContent"]
        self.assertEqual(result["request"]["output_format"], "jpeg")

    def test_explicit_delivery_resolution_remains_a_parameter(self) -> None:
        config = self.clean_config()
        response = self.call_image_tool(
            config,
            prompt="一张具备8K超高分辨率效果的产品图，输出分辨率为4k。",
            resolution="4k",
            resolution_user_requested=True,
        )
        self.assertNotIn("error", response)
        request = response["result"]["structuredContent"]["request"]
        self.assertEqual(request["size"], "2880x2880")

    def test_visual_resolution_language_is_documented_in_tool_schema(self) -> None:
        config = self.clean_config()
        tool = tool_map(call_server(config, "tools/list", {}))["generate_image"]
        resolution_description = tool["inputSchema"]["properties"]["resolution"]["description"]
        self.assertIn("8K超高分辨率", resolution_description)
        self.assertIn("do not map it here", resolution_description)

    def test_batch_route_keeps_each_prompt_direct(self) -> None:
        config = self.clean_config()
        response = call_server(
            config,
            "tools/call",
            {
                "name": "generate_image_batch",
                "arguments": {
                    "dry_run": True,
                    "verbose": True,
                    "jobs": [
                        {"prompt": "first complete prompt"},
                        {"prompt": "second complete prompt"},
                    ],
                },
            },
        )
        self.assertNotIn("error", response)
        result = response["result"]["structuredContent"]
        self.assertEqual(
            [job["prompt"] for job in result["request"]["jobs"]],
            ["first complete prompt", "second complete prompt"],
        )
        self.assertNotIn("prompt_preparations", result)

    def test_transparent_background_is_forwarded_for_standard_tools(self) -> None:
        config = self.clean_config()
        response = self.call_image_tool(
            config,
            prompt="transparent product cutout",
            background="transparent",
        )
        self.assertNotIn("error", response)
        request = response["result"]["structuredContent"]["request"]
        self.assertEqual(request["background"], "transparent")
        self.assertEqual(request["output_format"], "png")

        with tempfile.TemporaryDirectory() as temporary_directory:
            reference = Path(temporary_directory) / "reference.png"
            reference.write_bytes(TRANSPARENT_TEST_PNG)
            edit = self.call_image_tool(
                config,
                name="edit_image",
                prompt="turn the reference into a transparent product cutout",
                images=[str(reference)],
                background="transparent",
            )
            self.assertNotIn("error", edit)
            edit_request = edit["result"]["structuredContent"]["request"]
            self.assertEqual(edit_request["background"], "transparent")
            self.assertEqual(edit_request["output_format"], "png")

    def test_removed_prompt_preparation_fields_are_rejected(self) -> None:
        for field in ("compatibility", "compatibility_profile", "prompt_profile", "prompt_policy"):
            config = provider_config(
                ["legacy"],
                {"legacy": {**provider(), field: "obsolete"}},
            )
            response = self.call_image_tool(config, prompt="complete prompt")
            self.assertIn("error", response)
            self.assertIn("removed prompt-preparation fields", response["error"]["message"])
            self.assertIn(field, response["error"]["message"])


if __name__ == "__main__":
    unittest.main()
