#!/usr/bin/env python3
"""Apply the fixed Fmage image-regression pipeline in one local pass."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image, ImageChops, ImageFilter, ImageOps
except ModuleNotFoundError as error:
    if error.name == "PIL":
        raise SystemExit("Pillow is required: install the 'Pillow' package and run once again.") from error
    raise


MAX_PIXELS = 1_048_576
BLUR_SIGMA_PX = 1.2
NOISE_AMOUNT_PERCENT = 0.5
NOISE_SIGMA_LEVELS = 255.0 * NOISE_AMOUNT_PERCENT / 100.0


def target_size(width: int, height: int) -> tuple[int, int]:
    area = width * height
    if area <= MAX_PIXELS:
        return width, height

    scale = math.sqrt(MAX_PIXELS / area)
    target_width = max(1, math.floor(width * scale))
    target_height = max(1, math.floor(height * scale))

    if target_width * target_height > MAX_PIXELS:
        if target_width >= target_height:
            target_width = max(1, MAX_PIXELS // target_height)
        else:
            target_height = max(1, MAX_PIXELS // target_width)

    return target_width, target_height


def non_empty_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def effective_config_path(explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve(strict=False)

    fmage_config = non_empty_string(os.environ.get("FMAGE_CONFIG"))
    if fmage_config:
        return Path(fmage_config).expanduser().resolve(strict=False)

    antigravity_home = non_empty_string(os.environ.get("ANTIGRAVITY_HOME"))
    base_dir = (
        Path(antigravity_home).expanduser()
        if antigravity_home
        else Path.home() / ".gemini" / "antigravity"
    )
    return (base_dir / "fmage" / "providers.json").resolve(strict=False)


def load_output_config(config_path: Path) -> dict[str, object]:
    with config_path.open("r", encoding="utf-8-sig") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict) or not isinstance(config.get("providers"), dict):
        raise ValueError(f"Invalid Fmage configuration at {config_path}: missing providers object")
    return config


def first_active_provider(config: dict[str, object], config_path: Path) -> str:
    configured = config.get("active_providers")
    if isinstance(configured, list):
        providers = [non_empty_string(value) for value in configured]
    else:
        legacy = config.get("active_provider")
        providers = (
            [non_empty_string(value) for value in legacy]
            if isinstance(legacy, list)
            else [non_empty_string(legacy)]
        )

    provider_name = next((value for value in providers if value), None)
    if not provider_name:
        raise ValueError(f"No active_providers are configured in {config_path}")

    provider_map = config["providers"]
    if provider_name not in provider_map or not isinstance(provider_map[provider_name], dict):
        raise ValueError(f'Provider "{provider_name}" does not exist in the Fmage configuration')
    return provider_name


def resolve_config_directory(config_path: Path, value: object, fallback: Path) -> Path:
    configured = non_empty_string(value)
    if not configured:
        return fallback.resolve(strict=False)
    directory = Path(configured).expanduser()
    if not directory.is_absolute():
        directory = config_path.parent / directory
    return directory.resolve(strict=False)


def configured_output_root(
    config_path: Path, config: dict[str, object], provider_name: str
) -> Path:
    provider = config["providers"][provider_name]
    configured = non_empty_string(provider.get("output_dir")) or non_empty_string(
        config.get("output_dir")
    )
    fallback = config_path.parent / "outputs" / provider_name
    return resolve_config_directory(config_path, configured, fallback)


def unique_output_path(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    attempt = 0
    while True:
        suffix = "" if attempt == 0 else f"-{attempt}"
        candidate = output_dir / f"{timestamp}-001{suffix}.png"
        if not candidate.exists():
            return candidate
        attempt += 1


def add_monochrome_luminance_noise(image: Image.Image) -> Image.Image:
    has_alpha = image.mode == "RGBA"
    rgb = image.convert("RGB")
    alpha = image.getchannel("A") if has_alpha else None

    noise = Image.effect_noise(image.size, NOISE_SIGMA_LEVELS)
    shared_noise = Image.merge("RGB", (noise, noise, noise))
    noisy_rgb = ImageChops.add(rgb, shared_noise, scale=1.0, offset=-128)

    if alpha is None:
        return noisy_rgb
    red, green, blue = noisy_rgb.split()
    return Image.merge("RGBA", (red, green, blue, alpha))


def process_image(input_path: Path, output_path: Path) -> dict[str, object]:
    input_path = input_path.expanduser().resolve(strict=True)
    output_path = output_path.expanduser().resolve(strict=False)
    if output_path.suffix.lower() != ".png":
        raise ValueError("Output path must end in .png")

    with Image.open(input_path) as source:
        icc_profile = source.info.get("icc_profile")
        oriented = ImageOps.exif_transpose(source)
        oriented.load()

    original_width, original_height = oriented.size
    output_width, output_height = target_size(original_width, original_height)
    has_alpha = "A" in oriented.getbands() or (
        oriented.mode == "P" and "transparency" in oriented.info
    )
    working = oriented.convert("RGBA" if has_alpha else "RGB")

    resized = (output_width, output_height) != (original_width, original_height)
    if resized:
        working = working.resize((output_width, output_height), Image.Resampling.LANCZOS)

    blurred = working.filter(ImageFilter.GaussianBlur(radius=BLUR_SIGMA_PX))
    processed = add_monochrome_luminance_noise(blurred)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_options: dict[str, object] = {
        "format": "PNG",
        "optimize": False,
        "compress_level": 1,
    }
    if icc_profile:
        save_options["icc_profile"] = icc_profile
    processed.save(output_path, **save_options)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "original_size": [original_width, original_height],
        "output_size": [output_width, output_height],
        "resized": resized,
        "max_pixels": MAX_PIXELS,
        "blur_sigma_px": BLUR_SIGMA_PX,
        "noise_amount_percent": NOISE_AMOUNT_PERCENT,
        "noise_sigma_levels": NOISE_SIGMA_LEVELS,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shrink to at most 1K total pixels, blur, and add monochrome luminance noise."
    )
    parser.add_argument("--input", required=True, type=Path, help="Source raster image.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Explicit providers.json path; otherwise use FMAGE_CONFIG/ANTIGRAVITY_HOME defaults.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = effective_config_path(args.config)
        config = load_output_config(config_path)
        provider_name = first_active_provider(config, config_path)
        output_dir = configured_output_root(config_path, config, provider_name)
        output_path = unique_output_path(output_dir)
        result = process_image(args.input, output_path)
        result.update(
            {
                "config_path": str(config_path),
                "output_dir": str(output_dir),
                "active_provider": provider_name,
            }
        )
    except Exception as error:
        print(f"Fmage image regression failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
