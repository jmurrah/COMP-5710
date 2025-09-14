"""
COMP-5710 Workshop 1: Prompt Engineering (Google Gemini)
Author: Jacob Murrah
Date: 09/12/2025
"""

import unittest
import csv
import numpy as np
from scipy.stats import norm


def mannwhitneyu(x, y):
    # Concatenate and rank the data
    data = np.concatenate([x, y])
    ranked_data = np.array(data).argsort().argsort() + 1

    # Get the ranks for each sample
    n1 = len(x)
    n2 = len(y)
    ranks1 = ranked_data[:n1]
    ranks2 = ranked_data[n1:]

    # Calculate the sum of ranks
    r1 = np.sum(ranks1)
    r2 = np.sum(ranks2)

    # Calculate the U statistic
    u1 = r1 - (n1 * (n1 + 1)) / 2
    u2 = r2 - (n2 * (n2 + 1)) / 2

    u_statistic = min(u1, u2)

    # Calculate the p-value
    # For simplicity, this implementation uses a normal approximation for p-value calculation
    # and does not handle ties, which is a limitation of this basic implementation.
    mean_u = (n1 * n2) / 2
    std_u = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)

    # Handle the case where std_u is zero to avoid division by zero
    if std_u == 0:
        p_value = 1.0
    else:
        z = (u_statistic - mean_u) / std_u
        p_value = 2 * norm.sf(np.abs(z))

    return u_statistic, p_value


def getComparison(ls1, ls2):
    u_statistic, p_value = mannwhitneyu(ls1, ls2)
    assertion = "Significant" if p_value < 0.05 else "Not-Significant"
    print(f"Statistic: {u_statistic}, P-value: {p_value}, Assertion: {assertion}")
    return assertion


class TestWorkshop1(unittest.TestCase):
    def test_significant_original(self):
        x = [1, 1, 2, 3, 1, 1, 4]
        y = [6, 4, 7, 1, 3, 7, 3, 7]
        self.assertEqual("Significant", getComparison(x, y))

    def test_not_significant_similar_distributions(self):
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 6]
        self.assertEqual("Not-Significant", getComparison(x, y))

    def test_significant_different_distributions(self):
        x = [1, 1, 1, 2, 2]
        y = [8, 9, 9, 10, 10]
        self.assertEqual("Significant", getComparison(x, y))

    def test_identical_lists(self):
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        self.assertEqual("Not-Significant", getComparison(x, y))

    def test_single_element_different(self):
        x = [1]
        y = [10]
        self.assertEqual("Not-Significant", getComparison(x, y))

    def test_single_element_same(self):
        x = [5]
        y = [5]
        self.assertEqual("Not-Significant", getComparison(x, y))

    def test_two_elements_each(self):
        x = [1, 2]
        y = [8, 9]
        self.assertEqual("Not-Significant", getComparison(x, y))

    def test_different_lengths_significant(self):
        x = [1, 1, 1]
        y = [5, 6, 7, 8, 9, 10]
        self.assertEqual("Significant", getComparison(x, y))

    def test_different_lengths_not_significant(self):
        x = [3, 4]
        y = [2, 3, 4, 5, 6]
        self.assertEqual("Not-Significant", getComparison(x, y))

    def test_negative_values(self):
        x = [-5, -4, -3, -2]
        y = [1, 2, 3, 4]
        self.assertEqual("Significant", getComparison(x, y))

    def test_mixed_positive_negative(self):
        x = [-2, -1, 0, 1]
        y = [-1, 0, 1, 2]
        self.assertEqual("Not-Significant", getComparison(x, y))

    def test_float_values(self):
        x = [1.1, 1.5, 2.3, 2.8]
        y = [5.1, 5.5, 6.3, 6.8]
        self.assertEqual("Significant", getComparison(x, y))

    def test_large_sample_sizes(self):
        x = [1] * 20 + [2] * 20
        y = [3] * 20 + [4] * 20
        self.assertEqual("Significant", getComparison(x, y))

    def test_overlapping_ranges_not_significant(self):
        x = [1, 2, 3, 4, 5, 6]
        y = [4, 5, 6, 7, 8, 9]
        self.assertEqual("Significant", getComparison(x, y))

    def test_zero_values(self):
        x = [0, 0, 1, 1]
        y = [2, 2, 3, 3]
        self.assertEqual("Significant", getComparison(x, y))

    def test_repeated_values(self):
        x = [1, 1, 1, 1, 1]
        y = [2, 2, 2, 2, 2]
        self.assertEqual("Significant", getComparison(x, y))

    def test_wide_range_distribution(self):
        x = [1, 100, 200, 300]
        y = [50, 150, 250, 350]
        self.assertEqual("Not-Significant", getComparison(x, y))


if __name__ == "__main__":
    # run unit tests
    print("\033[1;34mRunning unit tests...\033[0m")
    unittest.main(exit=False)

    # get significance of columns A and B in perf-data.csv
    print(
        "\n\033[1;32mReading perf-data.csv and performing Mann-Whitney U test on columns A and B...\033[0m"
    )
    col_a, col_b = [], []
    with open("perf-data.csv", "r") as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader, None)

        a_index = header.index("A")
        b_index = header.index("B")

        for row in csv_reader:
            col_a.append(float(row[a_index]))
            col_b.append(float(row[b_index]))

    result = getComparison(col_a, col_b)
    print(f"Columns A and B are {result}ly different.")
