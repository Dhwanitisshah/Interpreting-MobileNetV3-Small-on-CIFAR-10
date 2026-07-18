from pathlib import Path
from typing import Any, Union

import yaml


class DotDict(dict):
    """Dict subclass with attribute-style ("dot") access, recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def load_config(path: Union[str, Path]) -> DotDict:
    """Load a YAML experiment config (see configs/*.yaml) as a dot-accessible dict."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return DotDict(raw)
