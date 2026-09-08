#!/usr/bin/env python3
"""Refresh and verify the local Fmage Antigravity native plugin and provider configuration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_NAME = "fmage"


def run_command(command: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PLUGIN_ROOT,
        check=False,
        capture_output=capture_output,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def require_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode == 0:
        return
    output = (result.stdout or "") + (result.stderr or "")
    detail = output.strip()
    raise RuntimeError(f"{label} failed{': ' + detail if detail else ''}")


def verify_plugin_manifest() -> dict[str, Any]:
    manifest_path = PLUGIN_ROOT / "plugin.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing Antigravity plugin manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON in plugin manifest: {manifest_path}") from error

    if not isinstance(manifest, dict):
        raise RuntimeError(f"Plugin manifest must be a JSON object: {manifest_path}")
    if manifest.get("name") != PLUGIN_NAME:
        raise RuntimeError(f"Plugin manifest 'name' must be '{PLUGIN_NAME}'; got {manifest.get('name')!r}")
    return manifest


def verify_mcp_config() -> dict[str, Any]:
    mcp_config_path = PLUGIN_ROOT / "mcp_config.json"
    if not mcp_config_path.is_file():
        raise RuntimeError(f"Missing MCP server configuration: {mcp_config_path}")
    try:
        config = json.loads(mcp_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON in mcp_config.json: {mcp_config_path}") from error

    servers = config.get("mcpServers")
    if not isinstance(servers, dict) or PLUGIN_NAME not in servers:
        raise RuntimeError(f"mcp_config.json must define mcpServers.{PLUGIN_NAME}")
    server_def = servers[PLUGIN_NAME]
    if not isinstance(server_def, dict) or not server_def.get("command"):
        raise RuntimeError(f"mcp_config.json mcpServers.{PLUGIN_NAME} must have a valid 'command'")
    return config


def verify_skills() -> list[str]:
    skills_dir = PLUGIN_ROOT / "skills"
    if not skills_dir.is_dir():
        raise RuntimeError(f"Missing skills directory: {skills_dir}")
    discovered: list[str] = []
    for skill_path in sorted(skills_dir.iterdir()):
        if not skill_path.is_dir():
            continue
        skill_md = skill_path / "SKILL.md"
        if not skill_md.is_file():
            continue
        content = skill_md.read_text(encoding="utf-8")
        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if not frontmatter_match:
            raise RuntimeError(f"Skill {skill_path.name} is missing YAML frontmatter in SKILL.md")
        fm = frontmatter_match.group(1)
        if not re.search(r"^name:\s*\S+", fm, re.MULTILINE):
            raise RuntimeError(f"Skill {skill_path.name} frontmatter is missing 'name' attribute")
        discovered.append(skill_path.name)
    if not discovered:
        raise RuntimeError(f"No valid skills found in {skills_dir}")
    return discovered


def verify_workspace_registration() -> None:
    plugins_json_path = PLUGIN_ROOT / ".agents" / "plugins.json"
    if not plugins_json_path.is_file():
        raise RuntimeError(f"Missing workspace plugin discovery file: {plugins_json_path}")
    try:
        config = json.loads(plugins_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON in .agents/plugins.json: {plugins_json_path}") from error
    entries = config.get("entries")
    if not isinstance(entries, list) or not any(
        isinstance(e, dict) and e.get("path") in {".", "./"} for e in entries
    ):
        raise RuntimeError(".agents/plugins.json must register the workspace root in 'entries'")


def test_mcp_server_syntax() -> None:
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required to run the Fmage MCP server.")
    server_mjs = PLUGIN_ROOT / "mcp" / "server.mjs"
    if not server_mjs.is_file():
        raise RuntimeError(f"Missing MCP server file: {server_mjs}")
    check = run_command([node, "--check", str(server_mjs)], capture_output=True)
    require_success(check, "Node syntax check on mcp/server.mjs")


def refresh_plugin(*, check_only: bool) -> None:
    manifest = verify_plugin_manifest()
    version = manifest.get("version", "unknown")
    verify_mcp_config()
    skills = verify_skills()
    verify_workspace_registration()
    test_mcp_server_syntax()

    status = "verified" if check_only else "refreshed and verified"
    print(f"Plugin {status}: {PLUGIN_NAME} v{version} (skills: {', '.join(skills)})")


def refresh_config(*, config: str | None, check_only: bool) -> None:
    command = [sys.executable, str(PLUGIN_ROOT / "scripts" / "migrate_local_config.py")]
    if config:
        command.extend(["--config", config])
    if check_only:
        command.append("--check")
    result = run_command(command)
    if check_only and result.returncode == 1:
        raise RuntimeError("Local Fmage configuration needs migration.")
    require_success(result, "migrating local Fmage configuration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to the local providers.json")
    parser.add_argument("--check", action="store_true", help="Only verify local runtime state")
    parser.add_argument("--skip-plugin", action="store_true")
    parser.add_argument("--skip-config", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.skip_config:
        refresh_config(config=args.config, check_only=args.check)
    if not args.skip_plugin:
        refresh_plugin(check_only=args.check)
    print("Fmage local runtime refresh complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
