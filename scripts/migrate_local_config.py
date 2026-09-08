#!/usr/bin/env python3
"""Migrate a local Fmage providers.json to the current provider schema."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path.home() / ".gemini" / "antigravity" / "fmage" / "providers.json"
OPENAI_IMAGES_TRANSPORT = "openai-images"
LEGACY_EZAI_NANO_TRANSPORT = "ezai-banana-images"
GEMINI_GENERATE_CONTENT_TRANSPORT = "gemini-generate-content"
EZAI_NANO_MODELS = {
    "nano-banana-2": "gemini-3.1-flash-image",
    "nano-banana-pro": "gemini-3-pro-image",
}
REMOVED_PROMPT_FIELDS = (
    "compatibility",
    "compatibility_profile",
    "prompt_profile",
    "prompt_policy",
)


def resolve_config_path(explicit: str | None = None) -> Path:
    if explicit and explicit.strip():
        return Path(explicit).expanduser().resolve()
    configured = os.environ.get("FMAGE_CONFIG", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    antigravity_home = os.environ.get("ANTIGRAVITY_HOME", "").strip()
    if antigravity_home:
        return (Path(antigravity_home).expanduser() / "fmage" / "providers.json").resolve()
    return DEFAULT_CONFIG_PATH.resolve()


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _is_removed_dalle_provider(provider_name: str, provider: dict[str, Any]) -> bool:
    identity = " ".join(
        str(value)
        for value in (provider_name, provider.get("model", ""))
        if value is not None
    ).lower()
    return bool(
        re.search(r"dall[\s_-]*(?:e|image)", identity)
        or re.search(r"(?:^|[^a-z])de3(?:[^a-z]|$)", identity)
    )


def _normalize_legacy_fields(provider: dict[str, Any], provider_name: str, changes: list[str]) -> None:
    if provider.get("transport") == "808-openai-images":
        provider["transport"] = OPENAI_IMAGES_TRANSPORT
        provider["transport_profile"] = "808"
        changes.append(f'{provider_name}: migrated transport to openai-images + profile 808')

    removed_fields = [field for field in REMOVED_PROMPT_FIELDS if field in provider]
    if removed_fields:
        for field in removed_fields:
            provider.pop(field, None)
        changes.append(
            f'{provider_name}: removed obsolete prompt-preparation fields ({", ".join(removed_fields)})'
        )

    model = provider.get("model")
    if provider.get("transport") == LEGACY_EZAI_NANO_TRANSPORT and model in EZAI_NANO_MODELS:
        provider["transport"] = GEMINI_GENERATE_CONTENT_TRANSPORT
        provider["model"] = EZAI_NANO_MODELS[model]
        provider.pop("response_format", None)
        changes.append(
            f'{provider_name}: migrated EzAI Nano to Gemini generateContent model {provider["model"]}'
        )


def migrate_payload(payload: dict[str, Any]) -> list[str]:
    providers = payload.get("providers")
    if not _is_mapping(providers):
        raise ValueError('Fmage configuration must contain an object named "providers".')

    changes: list[str] = []
    removed_provider_names: set[str] = set()
    for provider_name, provider in list(providers.items()):
        name = str(provider_name)
        if _is_mapping(provider) and _is_removed_dalle_provider(name, provider):
            providers.pop(provider_name, None)
            removed_provider_names.add(name)
            changes.append(f'removed provider "{name}"')
            continue
        if _is_mapping(provider):
            _normalize_legacy_fields(provider, name, changes)

    if providers.pop("808-MJ", None) is not None:
        changes.append('removed provider "808-MJ"')

    active_providers = payload.get("active_providers")
    if isinstance(active_providers, list):
        removed_active = [
            name
            for name in active_providers
            if name == "808-MJ" or name in removed_provider_names
        ]
        if removed_active:
            payload["active_providers"] = [
                name for name in active_providers if name not in set(removed_active)
            ]
            for name in removed_active:
                if name == "808-MJ":
                    changes.append('removed "808-MJ" from active_providers')
                else:
                    changes.append(f'removed "{name}" from active_providers')
    elif "active_providers" in payload:
        raise ValueError('Fmage configuration field "active_providers" must be an array.')

    active_provider = payload.get("active_provider")
    if isinstance(active_provider, str) and active_provider in removed_provider_names:
        payload["active_provider"] = None
        changes.append(f'removed "{active_provider}" from active_provider')

    return changes


def migrate_file(path: Path, *, check_only: bool = False) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Fmage configuration does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not _is_mapping(payload):
        raise ValueError("Fmage configuration must contain a JSON object.")
    changes = migrate_payload(payload)
    if changes and not check_only:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return changes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to the local providers.json")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report required migrations without writing the file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = resolve_config_path(args.config)
    changes = migrate_file(path, check_only=args.check)
    if not changes:
        print(f"Fmage local config is compatible: {path}")
        return 0
    action = "would migrate" if args.check else "migrated"
    print(f"Fmage local config {action}: {path}")
    for change in changes:
        print(f"- {change}")
    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
