"""Model configuration.

The model name has the PydanticAI form `provider:model`, for example
`google:gemini-3.1-flash-lite` or `ollama:llama3.2`. The caller loads `.env`
before it reads the configuration; this module only reads the environment.
"""

import os

DEFAULT_MODEL = 'google:gemini-3.1-flash-lite'


def model_name() -> str:
    """Return `GROUNDED_CV_MODEL`, or `DEFAULT_MODEL` if unset or empty."""
    return os.environ.get('GROUNDED_CV_MODEL') or DEFAULT_MODEL
