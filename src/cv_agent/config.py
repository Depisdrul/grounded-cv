"""Model configuration.

The model name has the PydanticAI form `provider:model`, for example
`google-gla:gemini-2.5-flash` or `ollama:llama3.2`. The caller loads `.env`
before it reads the configuration; this module only reads the environment.
"""

import os

DEFAULT_MODEL = "google-gla:gemini-2.5-flash"


def model_name() -> str:
    """Return `CV_AGENT_MODEL`, or `DEFAULT_MODEL` when it is not set or empty."""
    return os.environ.get("CV_AGENT_MODEL") or DEFAULT_MODEL
