import os
import unittest
from unittest import mock

from cv_agent.config import DEFAULT_MODEL, model_name


class ModelNameTest(unittest.TestCase):
    def test_returns_the_environment_value(self):
        with mock.patch.dict(os.environ, {"CV_AGENT_MODEL": "ollama:llama3.2"}, clear=True):
            self.assertEqual(model_name(), "ollama:llama3.2")

    def test_returns_the_default_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(model_name(), DEFAULT_MODEL)

    def test_returns_the_default_when_empty(self):
        with mock.patch.dict(os.environ, {"CV_AGENT_MODEL": ""}, clear=True):
            self.assertEqual(model_name(), DEFAULT_MODEL)


if __name__ == "__main__":
    unittest.main()
