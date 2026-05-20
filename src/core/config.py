"""配置加载模块 — 支持YAML文件 + 环境变量覆盖。"""
import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


def _resolve_env_vars(value: Any) -> Any:
    """递归解析字符串中的 ${ENV_VAR} 占位符。"""
    if isinstance(value, str):
        pattern = re.compile(r"\$\{(\w+)\}")
        matches = pattern.findall(value)
        for var in matches:
            env_val = os.environ.get(var, "")
            value = value.replace(f"${{{var}}}", env_val)
        return value
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


class Config:
    """全局配置单例，支持点号访问嵌套字典。"""

    _instance = None
    _data: dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        env = os.environ.get("RAG_ENV", "default")
        config_path = _CONFIG_DIR / f"{env}.yaml"
        if not config_path.exists():
            config_path = _CONFIG_DIR / "default.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f) or {}

        self._data = _resolve_env_vars(self._data)

    def get(self, key_path: str, default: Any = None) -> Any:
        """通过点号路径获取配置值，如 config.get('index.chunk_size')。"""
        keys = key_path.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT


config = Config()
