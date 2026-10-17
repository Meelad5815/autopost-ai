import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class AdminTerminalGateTests(unittest.TestCase):
    def test_terminal_gate_is_default_off(self):
        source = (ROOT / "app" / "routes" / "ui.py").read_text(encoding="utf-8")
        self.assertIn('os.getenv("ENABLE_ADMIN_TERMINAL", "").strip().lower()', source)
        self.assertIn('{"1", "true", "yes"}', source)
        self.assertIn("status_code=404", source)

    def test_both_command_execution_routes_use_gate(self):
        source = (ROOT / "app" / "routes" / "ui.py").read_text(encoding="utf-8")
        for function_name in ("cloud_terminal_exec", "kali_terminal_exec"):
            with self.subTest(function=function_name):
                start = source.index("def " + function_name + "(")
                next_def = source.find("\ndef ", start + 5)
                body = source[start: next_def if next_def != -1 else len(source)]
                self.assertIn("_require_terminal_enabled()", body)


if __name__ == "__main__":
    unittest.main()
