---
name: fmage-config
description: Locate, show, open, create, or edit the local Fmage providers.json. Use when explicitly invoked alone, when the user asks where the config is, or when updating active providers, transports, models, provider names, or provider keys.
---

# Fmage Config

- Reply in Chinese unless the user asks otherwise.
- If invoked with no extra request, do only the default action.

## Default action

Resolve the effective `providers.json` path and immediately reply with a clickable local file link plus the raw path. Do not say the skill is loaded.

Path resolution:
1. Use non-empty `FMAGE_CONFIG` exactly.
2. Else use non-empty `ANTIGRAVITY_HOME`.
3. Else use the current user's home directory plus `.gemini/antigravity`.
4. Unless step 1 was used, append `fmage/providers.json`.

Output:
providers.json: [providers.json](file:///ABSOLUTE_PATH_WITH_FORWARD_SLASHES)
路径: `ABSOLUTE_NATIVE_PATH`

Note: Always format the raw path `ABSOLUTE_NATIVE_PATH` as inline code with backticks (or in a code block) so Windows backslashes (e.g. `\.gemini`) are never swallowed or treated as Markdown escape characters. Use `file:///` with forward slashes for the clickable link.

If `FMAGE_CONFIG` was used, add one short note that it overrides the default.

## Edit rules

- Before editing, inspect the file and preserve existing providers.
- If missing and creation is requested, copy from plugin `config/providers.example.json`.
- Edit non-secret fields normally: `active_providers`, legacy `active_provider`, `transport`,
  `transport_profile`, `base_url`, `model`, `response_format`, `timeout`, and provider names.
  `transport: "808-openai-images"`, `compatibility`, `compatibility_profile`, `prompt_profile`,
  and `prompt_policy` are obsolete; remove them before use.
- The primary transports are `openai-images`, `ezai-banana-images`,
  `gemini-generate-content`, and `zenmux-vertex`. Use `gemini-generate-content` for providers that
  expose Google's native `/v1beta/models/{model}:generateContent` protocol; it supports generation
  and edits through text and inline image parts.
  Gemini authentication defaults to `x-goog-api-key`; the 808 profile uses the provider's
  documented `Authorization: Bearer` header internally, without an extra config field.
  `gemini-3.1-flash-image-preview` and `gemini-3.1-flash-image` share Nano Banana 2
  capabilities; `gemini-3-pro-image-preview` and `gemini-3-pro-image` share Pro capabilities.
  Keep the configured wire model ID unchanged. The `808-nano` example uses the Flash preview ID,
  `https://api.808relay.com`, `transport_profile: "808"`, and `timeout: 600`. The profile uses
  async submission and polling only for `openai-images`; Gemini keeps its native generateContent
  request/response flow while sharing the 808 authentication and timeout policy.
  `response_format: "url"`, and `timeout: 600`. Preserve the existing `api_key` or `api_key_env`.
- Providers use the normal direct prompt path. Do not configure prompt-profile or prompt-preparation fields.
- Never print existing API keys or ask the user to paste keys into chat; tell them to edit keys directly in `providers.json`.
