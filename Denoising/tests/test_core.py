import unittest

from src.metrics import mse, psnr, ssim


class TestCoreMetrics(unittest.TestCase):
    def test_metrics_basic_values(self):
        a = [[0.0, 1.0], [1.0, 0.0]]
        b = [[0.0, 1.0], [1.0, 0.0]]
        self.assertAlmostEqual(mse(a, b), 0.0, places=6)
        self.assertGreater(psnr(a, b), 100.0)
        self.assertAlmostEqual(ssim(a, b), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
