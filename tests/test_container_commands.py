import unittest
from container_cli import ROOT, resolve_command


class ContainerCommandTests(unittest.TestCase):
    def test_safe_default(self):
        path, args = resolve_command([])
        self.assertEqual((path.name, args), ("run.py", ["embedding"]))

    def test_all_resources_in_project(self):
        for command in ["embedding", "probe", "prepare", "index", "answer", "export", "evaluate", "sample", "reload", "smoke"]:
            path, _ = resolve_command([command])
            self.assertTrue(path.is_file(), str(path))
            self.assertTrue(path.is_relative_to(ROOT))

    def test_evaluation_arguments_forwarded(self):
        path, args = resolve_command(["evaluate", "--input", "/app/outputs/eval.json"])
        self.assertEqual(path, ROOT / "evaluation/run.py")
        self.assertEqual(args, ["--input", "/app/outputs/eval.json"])

    def test_invalid_command_does_not_execute_arbitrary_path(self):
        with self.assertRaises(ValueError):
            resolve_command(["../../anything.py"])


if __name__ == "__main__":
    unittest.main()
