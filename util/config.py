import tomllib
from typing import Any, Dict


class ConfigManager:
    config: Dict[str, Any] = {}

    @classmethod
    def load(cls, path: str) -> None:
        with open(path, "rb") as f:
            cls.config = tomllib.load(f)

    @classmethod
    def get(cls) -> Dict[str, Any]:
        return cls.config
