import os
import unittest
from unittest import mock

from grounded_cv.config import product_model_name


class ProductModelNameTest(unittest.TestCase):
    def test_returns_the_environment_value(self):
        with mock.patch.dict(
            os.environ, {'GROUNDED_CV_MODEL': 'ollama:llama3.2'}, clear=True
        ):
            self.assertEqual(product_model_name(), 'ollama:llama3.2')

    def test_returns_the_local_model_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(product_model_name(), 'ollama:qwen3.5-4b-16k')

    def test_returns_the_local_model_when_empty(self):
        with mock.patch.dict(os.environ, {'GROUNDED_CV_MODEL': ''}, clear=True):
            self.assertEqual(product_model_name(), 'ollama:qwen3.5-4b-16k')


if __name__ == '__main__':
    unittest.main()
