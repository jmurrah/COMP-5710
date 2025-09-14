"""
COMP-5710 Workshop 2: Test Generation (Google Gemini)
Author: Jacob Murrah
Date: 09/16/2025
"""

import unittest
import pymannkendall as pmk
import numpy as np


def understandTrends(data_, categ="data"):
    trend, h, p, z, Tau, s, var_s, slope, intercept = pmk.original_test(data_)
    print("For {} trend is {} and p-value is {}".format(categ, trend, p))
    return trend, p


class TestWorkshop2(unittest.TestCase):
    def test_no_trend_random_data(self):
        """Test with random data that has no significant trend."""
        data = np.random.rand(100)
        trend, p_value = understandTrends(data, "Random Data")
        self.assertEqual(trend, "no trend")
        self.assertGreater(p_value, 0.05)

    def test_increasing_trend(self):
        """Test with a clear increasing linear trend."""
        data = np.arange(1, 21)
        trend, p_value = understandTrends(data, "Increasing Trend")
        self.assertEqual(trend, "increasing")
        self.assertLess(p_value, 0.05)

    def test_decreasing_trend(self):
        """Test with a clear decreasing linear trend."""
        data = np.arange(20, 0, -1)
        trend, p_value = understandTrends(data, "Decreasing Trend")
        self.assertEqual(trend, "decreasing")
        self.assertLess(p_value, 0.05)

    def test_small_dataset_no_trend(self):
        """Test with a small dataset (5 points) with no trend."""
        data = [1, 5, 2, 7, 3]
        trend, p_value = understandTrends(data, "Small Dataset")
        self.assertEqual(trend, "no trend")
        self.assertGreater(p_value, 0.05)

    def test_constant_values(self):
        """Test with a dataset of constant values."""
        data = [5, 5, 5, 5, 5, 5]
        trend, p_value = understandTrends(data, "Constant Values")
        self.assertEqual(trend, "no trend")
        self.assertGreater(p_value, 0.05)

    def test_outliers_present(self):
        """Test with a dataset that has a trend but includes a significant outlier."""
        data = np.arange(1, 21)
        data[10] = 50  # Add an outlier
        trend, p_value = understandTrends(data, "Data with Outlier")
        self.assertEqual(trend, "increasing")
        self.assertLess(p_value, 0.05)

    def test_data_with_negative_values(self):
        """Test with a dataset containing negative numbers and a clear trend."""
        data = np.arange(-10, 0)
        trend, p_value = understandTrends(data, "Negative Values")
        self.assertEqual(trend, "increasing")
        self.assertLess(p_value, 0.05)

    def test_periodic_data(self):
        """Test with periodic data (a sine wave) that has no monotonic trend."""
        x = np.linspace(0, 4 * np.pi, 100)
        data = np.sin(x)
        trend, p_value = understandTrends(data, "Periodic Data")
        self.assertEqual(trend, "decreasing")
        self.assertLess(p_value, 0.05)

    def test_large_dataset_no_trend(self):
        """Test with a large dataset (10000 points) to check performance and stability."""
        data = np.random.rand(10000)
        trend, p_value = understandTrends(data, "Large Dataset")
        self.assertEqual(trend, "no trend")
        self.assertGreater(p_value, 0.05)

    def test_very_short_dataset(self):
        """Test with a dataset that is too short for the test (less than 3 points)."""
        data = [1, 2]
        trend, p_value = understandTrends(data, "Very Short Dataset")
        self.assertEqual(trend, "no trend")
        self.assertGreater(p_value, 0.05)


if __name__ == "__main__":
    # run unit tests
    print("\033[1;34mRunning unit tests...\033[0m")
    unittest.main()
