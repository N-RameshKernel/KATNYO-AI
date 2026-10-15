import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from services.ai_provider import _provider_completion


class ProviderHttpErrorTests(unittest.TestCase):
    def provider_error(self, status_code):
        error = HTTPError("https://api.example.test", status_code, "failed", {}, None)
        with patch("services.ai_provider.urlopen", side_effect=error):
            with self.assertRaises(RuntimeError) as raised:
                _provider_completion("openai", "test", {"api_key": "secret-test-key"})
        self.assertNotIn("secret-test-key", str(raised.exception))
        return str(raised.exception)

    def test_unauthorized_message_suggests_replacing_credential_safely(self):
        message = self.provider_error(401)
        self.assertIn("rejected the API credential", message)
        self.assertIn("correct provider/project", message)
        self.assertIn("Save & test", message)

    def test_permission_not_found_and_quota_errors_are_distinguished(self):
        self.assertIn("denied access", self.provider_error(403))
        self.assertIn("requested model or endpoint", self.provider_error(404))
        self.assertIn("quota", self.provider_error(429))


if __name__ == "__main__":
    unittest.main()
