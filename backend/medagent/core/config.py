import os
from dataclasses import dataclass, fields
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
# Connection details and secrets stay out of the settings hash stored in reports.
UNHASHED_SETTINGS = {"api_key", "api_url", "fallback_api_key", "fallback_api_url", "redis_url", "chroma_host", "chroma_port", "skill_directory",
                     "appointment_db"}


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    model: str = ""
    api_url: str = "https://api.anthropic.com"
    # Vendors whose Anthropic-format endpoint has no output_config JSON schema get the schema in the system prompt.
    structured_output: bool = True
    thinking: bool = False
    # Standby role instance at another vendor; an empty key keeps both instances on the primary model.
    fallback_api_key: str = ""
    fallback_model: str = "deepseek-flash"
    fallback_api_url: str = "https://api.deepseek.com/anthropic"
    # Judge on the same vendor as the standby; empty keeps it on the primary model.
    judge_model: str = ""
    redis_url: str = "redis://localhost:6379/0"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    preview: bool = False
    embeddings: str = "onnx"
    emergency_phone: str = "当地急救电话"
    model_timeout: float = 35
    agent_timeout: float = 100
    tool_timeout: float = 20
    tool_call_limit: int = 12
    memory_timeout: float = 20
    episode_ttl: float = 7*86400
    supporting_agent_limit: int = 2
    # "single_agent" is the comparison baseline only; the workbench and default runs use "multi_agent".
    architecture: str = "multi_agent"
    # Comparison only: the single agent holds every booking tool on every turn, with no request admission.
    single_agent_open_tools: bool = False
    single_agent_attempts: int = 1
    agent_concurrency: int = 2
    collaboration_timeout: float = 120
    monitor_interval: float = 10
    monitor_window_size: int = 40
    monitor_window_seconds: float = 300
    monitor_min_samples: int = 5
    monitor_agent_success_rate: float = .90
    monitor_agent_latency_ms: float = 15000
    monitor_tool_success_rate: float = .90
    monitor_tool_latency_ms: float = 10000
    monitor_local_tool_latency_ms: float = 1000
    monitor_penalty_step: float = .25
    monitor_max_penalty: float = .8
    monitor_fallback_penalty: float = .5
    monitor_bad_ticks: int = 2
    monitor_recovery_successes: int = 2
    monitor_recovery_ticks: int = 2
    monitor_probe_cooldown: float = 30
    monitor_probe_timeout: float = 15
    skill_directory: Path = ROOT / "resources" / "skills"
    # Mock hospital only. In memory by default, so nothing to commit and a restart clears it.
    appointment_db: str = ":memory:"
    appointment_ttl: float = 600
    appointment_followup_ttl: float = 600
    appointment_timezone: str = "Asia/Shanghai"

    @classmethod
    def load(cls):
        load_dotenv(ROOT / ".env.local", override=False)
        env = os.environ
        return cls(
            api_key=env.get("ANTHROPIC_API_KEY", ""),
            model=env.get("ANTHROPIC_MODEL", ""),
            api_url=env.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/"),
            fallback_api_key=env.get("DEEPSEEK_API_KEY", ""),
            fallback_model=env.get("DEEPSEEK_MODEL", "deepseek-flash"),
            fallback_api_url=env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/anthropic").rstrip("/"),
            judge_model=env.get("DEEPSEEK_JUDGE_MODEL", ""),
            redis_url=env.get("REDIS_URL", "redis://localhost:6379/0"),
            chroma_host=env.get("CHROMA_HOST", "localhost"),
            chroma_port=int(env.get("CHROMA_PORT", "8001")),
            preview=env.get("MAR_PREVIEW", "false").lower() == "true",
            embeddings=env.get("MAR_EMBEDDINGS", "onnx"),
            emergency_phone=env.get("MAR_EMERGENCY_PHONE", "当地急救电话"),
            model_timeout=float(env.get("MAR_MODEL_TIMEOUT", "35")),
            agent_timeout=float(env.get("MAR_AGENT_TIMEOUT", "100")),
            tool_timeout=float(env.get("MAR_TOOL_TIMEOUT", "20")),
            tool_call_limit=int(env.get("MAR_TOOL_CALL_LIMIT", "12")),
            memory_timeout=float(env.get("MAR_MEMORY_TIMEOUT", "20")),
            episode_ttl=float(env.get("MAR_EPISODE_TTL_SECONDS", str(7*86400))),
            supporting_agent_limit=int(env.get("MAR_SUPPORTING_AGENT_LIMIT", "2")),
            architecture=env.get("MAR_ARCHITECTURE", "multi_agent"),
            single_agent_attempts=int(env.get("MAR_SINGLE_AGENT_ATTEMPTS", "1")),
            agent_concurrency=int(env.get("MAR_AGENT_CONCURRENCY", "2")),
            collaboration_timeout=float(env.get("MAR_COLLABORATION_TIMEOUT", "120")),
            # The app keeps bookings across restarts in an ignored local file; tests use Settings() and stay in memory.
            appointment_db=env.get("MAR_APPOINTMENT_DB", str(ROOT / "data" / "appointments.sqlite3")),
            appointment_followup_ttl=float(env.get("MAR_APPOINTMENT_FOLLOWUP_TTL", "600")),
            appointment_timezone=env.get("MAR_APPOINTMENT_TIMEZONE", "Asia/Shanghai"),
            **{field.name:(int if field.name in {"monitor_window_size", "monitor_min_samples", "monitor_bad_ticks", "monitor_recovery_successes", "monitor_recovery_ticks"} else float)
               (env.get("MAR_"+field.name.upper(), str(field.default))) for field in fields(cls) if field.name.startswith("monitor_")},
        )
