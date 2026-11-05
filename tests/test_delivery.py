import unittest
from pathlib import Path
from tools.prepare_delivery import delivery_files, excluded, ROOT


class DeliveryTests(unittest.TestCase):
    def test_secret_files_excluded(self):
        self.assertTrue(excluded(Path(".env")))
        self.assertTrue(excluded(Path("sub/.env.production")))
        self.assertFalse(excluded(Path(".env.example")))

    def test_cache_and_outputs_excluded(self):
        for name in ["cache/wheels/torch.whl", "outputs/predictions.json", "vendor/HippoRAG/.git/config", "evaluation/__pycache__/metrics.pyc"]:
            self.assertTrue(excluded(Path(name)))

    def test_runtime_assets_in_delivery(self):
        paths = {p.relative_to(ROOT).as_posix() for p in delivery_files()}
        for name in ["Dockerfile", "compose.server-build.yaml", "evaluation/config.json", "models/bge-m3/pytorch_model.bin", "data/banking/银行监管测试集/questions.json", "vendor/HippoRAG/pyproject.toml"]:
            self.assertIn(name, paths)
        self.assertNotIn(".env", paths)
        self.assertFalse(any(name.startswith("outputs/") or name.startswith("cache/") for name in paths))


if __name__ == "__main__":
    unittest.main()
