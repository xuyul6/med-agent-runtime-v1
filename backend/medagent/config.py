import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    model: str = ""
    api_url: str = "https://api.anthropic.com"
    redis_url: str = "redis://localhost:6379/0"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    preview: bool = False
    embeddings: str = "onnx"
    emergency_phone: str = "当地急救电话"
    model_timeout: float = 35
    agent_timeout: float = 100
    tool_timeout: float = 20
    memory_timeout: float = 20
    skill_directory: Path = ROOT / "resources" / "skills"

    @classmethod
    def load(cls):
        load_dotenv(ROOT / ".env.local", override=False)
        env = os.environ
        return cls(
            api_key=env.get("ANTHROPIC_API_KEY", ""),
            model=env.get("ANTHROPIC_MODEL", ""),
            api_url=env.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/"),
            redis_url=env.get("REDIS_URL", "redis://localhost:6379/0"),
            chroma_host=env.get("CHROMA_HOST", "localhost"),
            chroma_port=int(env.get("CHROMA_PORT", "8001")),
            preview=env.get("MAR_PREVIEW", "false").lower() == "true",
            embeddings=env.get("MAR_EMBEDDINGS", "onnx"),
            emergency_phone=env.get("MAR_EMERGENCY_PHONE", "当地急救电话"),
            model_timeout=float(env.get("MAR_MODEL_TIMEOUT", "35")),
            agent_timeout=float(env.get("MAR_AGENT_TIMEOUT", "100")),
            tool_timeout=float(env.get("MAR_TOOL_TIMEOUT", "20")),
            memory_timeout=float(env.get("MAR_MEMORY_TIMEOUT", "20")),
        )
