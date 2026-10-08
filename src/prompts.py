"""Load the prompt file and render templates."""
from pathlib import Path
from string import Template

import yaml


def load_prompts(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def render(prompts: dict, task: str, **values) -> tuple[str, str]:
    """Return (system, user) for a task, filling $placeholders in the user prompt."""
    spec = prompts[task]
    return spec["system"], Template(spec["user"]).safe_substitute(**values)
