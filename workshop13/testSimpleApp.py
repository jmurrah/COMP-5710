"""
COMP-5710 Workshop 13: Functional Testing (test file)
Author: Jacob Murrah
Date: 12/2/2025
"""

import unittest
from unittest.mock import patch
from simpleApp import app
import sys

EXPECTED_WELCOME = "Welcome to a Simple Flask API!"
EXPECTED_SQA = "Welcome to the SQA course!"
EXPECTED_SSP = "Secure Software Process"
EXPECTED_VANITY = "Jacob Murrah"
EXPECTED_MYPYTHON = sys.version
EXPECTED_CSSE = "Department of Computer Science and Software Engineering"


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    # helper methods
    def to_h1_bytes(self, message):
        return f"<h1>{message}</h1>".encode()

    def assert_get_matches(self, endpoint, expected_message):
        response = self.client.get(endpoint)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, self.to_h1_bytes(expected_message))

    def assert_only_get_allowed(self, endpoint):
        for method in ["POST", "PUT", "DELETE", "PATCH"]:
            with self.subTest(endpoint=endpoint, method=method):
                response = self.client.open(endpoint, method=method)
                self.assertEqual(response.status_code, 405)

    def assert_incorrect_message_rejected(
        self, endpoint, const_name, wrong_message, expected_message
    ):
        with patch(f"simpleApp.{const_name}", wrong_message):
            response = self.client.get(endpoint)
        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(response.data, self.to_h1_bytes(expected_message))

    # test methods
    def test_home_get(self):
        self.assert_get_matches("/", EXPECTED_WELCOME)

    def test_home_only_get_allowed(self):
        self.assert_only_get_allowed("/")

    def test_home_incorrect_message_rejected(self):
        self.assert_incorrect_message_rejected(
            "/", "WELCOME_MESSAGE", "Wrong", EXPECTED_WELCOME
        )

    def test_sqa_get(self):
        self.assert_get_matches("/sqa", EXPECTED_SQA)

    def test_sqa_only_get_allowed(self):
        self.assert_only_get_allowed("/sqa")

    def test_sqa_incorrect_message_rejected(self):
        self.assert_incorrect_message_rejected(
            "/sqa", "SQA_MESSAGE", "Not SQA", EXPECTED_SQA
        )

    def test_ssp_get(self):
        self.assert_get_matches("/ssp", EXPECTED_SSP)

    def test_ssp_only_get_allowed(self):
        self.assert_only_get_allowed("/ssp")

    def test_ssp_incorrect_message_rejected(self):
        self.assert_incorrect_message_rejected(
            "/ssp", "SSP_MESSAGE", "Wrong SSP Message", EXPECTED_SSP
        )

    def test_vanity_get(self):
        self.assert_get_matches("/vanity", EXPECTED_VANITY)

    def test_vanity_only_get_allowed(self):
        self.assert_only_get_allowed("/vanity")

    def test_vanity_incorrect_message_rejected(self):
        self.assert_incorrect_message_rejected(
            "/vanity", "VANITY_MESSAGE", "Not Jacob", EXPECTED_VANITY
        )

    def test_mypython_get(self):
        self.assert_get_matches("/mypython", EXPECTED_MYPYTHON)

    def test_mypython_only_get_allowed(self):
        self.assert_only_get_allowed("/mypython")

    def test_mypython_incorrect_message_rejected(self):
        self.assert_incorrect_message_rejected(
            "/mypython", "MYPYTHON_MESSAGE", "Python 0.0", EXPECTED_MYPYTHON
        )

    def test_csse_get(self):
        self.assert_get_matches("/csse", EXPECTED_CSSE)

    def test_csse_only_get_allowed(self):
        self.assert_only_get_allowed("/csse")

    def test_csse_incorrect_message_rejected(self):
        self.assert_incorrect_message_rejected(
            "/csse", "CSSE_MESSAGE", "Wrong Dept", EXPECTED_CSSE
        )


if __name__ == "__main__":
    unittest.main()
