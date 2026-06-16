from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .schemas import LocalConfig, SaveConfigRequest, SecretConfig, model_to_dict

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG_PATH = DATA_DIR / "local_config.json"

SECRET_FIELDS = set(SecretConfig.model_fields.keys() if hasattr(SecretConfig, "model_fields") else SecretConfig.__fields__.keys())


def load_local_config() -> LocalConfig:
    if not CONFIG_PATH.exists():
        return LocalConfig()
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        config = LocalConfig(**raw)
        if config.benchmark.prompt_detail_level == "compact" and config.benchmark.max_output_tokens > 900 and not config.benchmark.no_paid_api_mode:
            config.benchmark.max_output_tokens = 900
        if config.benchmark.prompt_detail_level == "compact" and config.benchmark.max_news_per_symbol > 2:
            config.benchmark.max_news_per_symbol = 2
        return config
    except Exception:
        return LocalConfig()


def save_local_config(payload: SaveConfigRequest) -> LocalConfig:
    current = load_local_config()
    incoming = LocalConfig(benchmark=payload.benchmark, secrets=payload.secrets)

    current_secrets = model_to_dict(current.secrets)
    incoming_secrets = model_to_dict(incoming.secrets)
    merged_secrets: Dict[str, Any] = {}
    for key in SECRET_FIELDS:
        new_value = incoming_secrets.get(key, "")
        if new_value:
            merged_secrets[key] = new_value
        else:
            merged_secrets[key] = current_secrets.get(key, "")

    saved = LocalConfig(benchmark=incoming.benchmark, secrets=SecretConfig(**merged_secrets))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(model_to_dict(saved), indent=2), encoding="utf-8")
    return saved


def public_config(config: LocalConfig | None = None) -> Dict[str, Any]:
    config = config or load_local_config()
    secrets = model_to_dict(config.secrets)
    secret_status = {key: bool(value) for key, value in secrets.items()}
    public_secrets = {key: "" for key in secrets}
    public_secrets["openai_base_url"] = secrets.get("openai_base_url") or "https://api.openai.com/v1"
    return {
        "benchmark": model_to_dict(config.benchmark),
        "secrets": public_secrets,
        "secret_status": secret_status,
        "config_path": str(CONFIG_PATH),
    }
