#!/usr/bin/env node
/**
 * Antigravity PreToolUse hook handler for intercepting native image generation calls.
 * Gated tools: generate_image
 */
import readline from 'node:readline';

async function main() {
  let input = '';
  const rl = readline.createInterface({ input: process.stdin });

  for await (const line of rl) {
    input += line + '\n';
  }

  let payload = {};
  if (input.trim()) {
    try {
      payload = JSON.parse(input);
    } catch {
      // Ignore payload parsing errors and proceed with default deny response
    }
  }

  const toolName = payload?.toolCall?.name || '';
  const reason =
    'Antigravity native image generation tool is disabled by Fmage plugin policy. ' +
    'Use Fmage MCP tools instead (fmage.generate_image, fmage.edit_image, fmage.generate_image_batch, or fmage.edit_image_batch).';

  const response = {
    decision: 'deny',
    reason,
  };

  process.stdout.write(JSON.stringify(response) + '\n');
}

main().catch((err) => {
  process.stderr.write(String(err) + '\n');
  process.stdout.write(JSON.stringify({ decision: 'deny', reason: 'Hook execution failed; blocking tool by default.' }) + '\n');
  process.exit(0);
});
