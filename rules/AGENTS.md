# Fmage Plugin Rules

## Image Generation And Editing

- For raster image generation, image editing, color correction, color optimization, retouching, background changes, style transfer, or any other task that produces or edits an image, use the Fmage plugin.
- Load and follow the `fmage` skill, then call the Fmage MCP tools (`generate_image`, `edit_image`, `generate_image_batch`, or `edit_image_batch`) provided by the `fmage` server.
- Do not use Antigravity native `generate_image` tool, the conversation artifacts directory, or any generic image generation skills for these tasks.
- Do not use Shell/Python/PIL/ImageMagick scripts as a fallback for image generation or image editing unless the user explicitly asks for deterministic local pixel processing.
- If Fmage tools are not visible, first run proper tool discovery and `get_provider_status`. If the tools still cannot be called, report the blocker instead of silently switching to another image tool or local script.
- After any Fmage image call fails or returns partial results, stop and report the failure. Do not retry, change parameters, query providers to find an alternative, or switch providers unless the user explicitly asks or approves it.
- Treat Fmage `1k`/`2k`/`3k`/`4k` values as resolution tiers. For `openai-images` and `json-images`, the normalized `requested_size` is authoritative; when actual dimensions match it and no size warning is present, do not describe the output as undersized or compare it with `4096x4096`.
