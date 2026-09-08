import json
from pathlib import Path
import subprocess
import unittest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


class TestPluginRulesAndHooks(unittest.TestCase):
    def test_rules_file_exists_and_contains_expected_policies(self):
        rule_path = PLUGIN_ROOT / 'rules' / 'AGENTS.md'
        self.assertTrue(rule_path.is_file(), 'rules/AGENTS.md should exist')
        content = rule_path.read_text(encoding='utf-8')
        self.assertIn('Fmage', content)
        self.assertIn('generate_image', content)
        self.assertIn('fmage', content)
        self.assertIn('requested_size', content)

    def test_hooks_json_structure(self):
        hooks_path = PLUGIN_ROOT / 'hooks.json'
        self.assertTrue(hooks_path.is_file(), 'hooks.json should exist')
        data = json.loads(hooks_path.read_text(encoding='utf-8'))
        self.assertIn('fmage-intercept-native-image', data)
        pre_tool_use = data['fmage-intercept-native-image'].get('PreToolUse')
        self.assertIsInstance(pre_tool_use, list)
        self.assertEqual(pre_tool_use[0].get('matcher'), 'generate_image')

    def test_hook_script_intercepts_native_image_call(self):
        script_path = PLUGIN_ROOT / 'scripts' / 'intercept_native_image.mjs'
        self.assertTrue(script_path.is_file())
        fake_payload = {
            'conversationId': 'test-convo-id',
            'toolCall': {
                'name': 'generate_image',
                'args': {
                    'Prompt': 'A scenic mountain landscape',
                    'ImageName': 'mountain',
                },
            },
            'stepIdx': 1,
        }
        proc = subprocess.run(
            ['node', str(script_path)],
            input=json.dumps(fake_payload),
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=False,
        )
        self.assertEqual(proc.returncode, 0, f'Hook script failed: {proc.stderr}')
        result = json.loads(proc.stdout.strip())
        self.assertEqual(result.get('decision'), 'deny')
        self.assertIn('Fmage', result.get('reason', ''))


if __name__ == '__main__':
    unittest.main()
