"""Agent instructions, one Markdown file per agent."""

from importlib.resources import files


def load_prompt(name: str) -> str:
    """Return the text of `prompts/<name>.md`.

    The prompts name code identifiers: the save_fact tool and the Status and
    Actor values. tests/test_prompts.py checks that they still match.
    """
    return files(__package__).joinpath(f'{name}.md').read_text(encoding='utf-8')
