"""Model configuration.

The model name has the PydanticAI form `provider:model`, for example
`ollama:qwen3.5-4b-16k` or `google:gemini-3.1-flash-lite`. The caller loads
`.env` before it reads the configuration; this module only reads the
environment.
"""

import os

DEFAULT_PRODUCT_MODEL = 'ollama:qwen3.5-4b-16k'
DEFAULT_INTERVIEWEE_AND_JUDGE_MODEL = 'google:gemini-3.1-flash-lite'


def product_model_name() -> str:
    """Return the model of the interviewer and the writer.

    Returns:
        `GROUNDED_CV_MODEL`, or `DEFAULT_PRODUCT_MODEL` if unset or empty.
    """
    return os.environ.get('GROUNDED_CV_MODEL') or DEFAULT_PRODUCT_MODEL
