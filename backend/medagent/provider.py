import json
import time
import httpx
from .observability import event
from .profiles import DEFAULT_REGISTRY
from .conversation import FIELD_NAMES


def role_json_schema(role: str) -> dict:
    profile = DEFAULT_REGISTRY[role]
    if profile.deterministic:
        raise ValueError("deterministic_role_has_no_model_schema")
    def normalize(node):
        if isinstance(node, list):
            return [normalize(item) for item in node]
        if not isinstance(node, dict):
            return node
        # Length/range constraints remain enforced locally; the provider gets the strict structural subset.
        result = {key:normalize(value) for key, value in node.items()
                  if key not in {"title", "default", "minLength", "maxLength", "minItems", "maxItems", "minimum", "maximum"}}
        if "const" in result:
            result["enum"] = [result.pop("const")]
        if result.get("type") == "object":
            result["additionalProperties"] = False
            result["required"] = list(result.get("properties", {}))
        return result
    return normalize(profile.output_contract.model_json_schema())


def json_schema(stage: str) -> dict | None:
    if stage.startswith("agent_"):
        return role_json_schema(stage.removeprefix("agent_"))
    string = {"type":"string"}
    arrays = {"type":"array", "items":string}
    properties = {
        "understanding": {
            "topic_relation":{"type":"string", "enum":["new", "followup", "uncertain"]},
            "scores":{"type":"object","properties":{key:{"type":"number"} for key in ["symptoms","medicine","visit","report","general"]},
                      "required":["symptoms","medicine","visit","report","general"],"additionalProperties":False},
            "tasks":{"type":"array", "items":{"type":"object", "properties":{"intent":{"type":"string", "enum":["symptoms","medicine","visit","report","general"]}, "goal":string},
                        "required":["intent", "goal"], "additionalProperties":False}},
            "updates":{"type":"array", "items":{"type":"object", "properties":{
                "name":{"type":"string", "enum":list(FIELD_NAMES)}, "value":{"anyOf":[string,{"type":"null"}]},
                "state":{"type":"string", "enum":["provided","topic","negated","unknown"]}, "quote":string, "turn_id":string},
                "required":["name","value","state","quote","turn_id"], "additionalProperties":False}},
        },
        "retrieval_rewrite": {"query":string},
        "retrieval_rerank": {"keys":arrays},
        "memory_summary": {"summary":string},
        "evaluation_judge": {**{key:{"type":"number"} for key in ["relevance","accuracy","completeness","helpfulness"]},"reason":string},
    }
    if stage == "synthesis":
        fields = {"section_ids":arrays}
    else:
        fields = properties.get(stage)
    return {"type":"object","properties":fields,"required":list(fields),"additionalProperties":False} if fields else None


class ModelFailure(RuntimeError):
    pass


def object_from_text(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = "\n".join(stripped.splitlines()[1:-1])
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


class ClaudeGateway:
    def __init__(self, settings, client=None):
        self.settings = settings
        self.client = client or httpx.AsyncClient(timeout=settings.model_timeout)

    async def message(self, stage: str, system: str, messages: list[dict], tools: list[dict] | None=None, budget: int=1800) -> dict:
        if not self.settings.api_key or not self.settings.model:
            raise ModelFailure("model_not_configured")
        payload = {"model": self.settings.model, "max_tokens": budget,
                   "system": system, "messages": messages, "thinking": {"type": "disabled"}}
        schema = json_schema(stage)
        if schema:
            payload["output_config"] = {"format":{"type":"json_schema","schema":schema}}
        if tools:
            payload["tools"] = tools
        try:
            started = time.perf_counter()
            response = await self.client.post(self.settings.api_url + "/v1/messages", json=payload,
                headers={"x-api-key": self.settings.api_key, "anthropic-version": "2023-06-01"})
            response.raise_for_status()
            result = response.json()
            usage = result.get("usage", {})
            event("model", stage=stage, model=result.get("model", self.settings.model),
                  response_model=result.get("model"),
                  elapsed_ms=round((time.perf_counter()-started)*1000,1),
                  input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                  cache_read_input_tokens=usage.get("cache_read_input_tokens"),
                  cache_creation_input_tokens=usage.get("cache_creation_input_tokens"),
                  stop=result.get("stop_reason"))
            if result.get("stop_reason") == "max_tokens":
                raise ModelFailure("model_output_truncated")
            return result
        except (httpx.HTTPError, ValueError) as exc:
            status = exc.response.status_code if isinstance(exc,httpx.HTTPStatusError) else None
            event("model_error", stage=stage, error=type(exc).__name__, http_status=status,
                  elapsed_ms=round((time.perf_counter()-started)*1000,1))
            raise ModelFailure("model_request_failed") from exc

    async def json(self, stage: str, system: str, content: str, budget: int=900) -> dict:
        result = await self.message(stage, system + "\n仅输出一个合法JSON对象，不使用Markdown。",
                                    [{"role": "user", "content": content}], budget=budget)
        try:
            return object_from_text("".join(x.get("text", "") for x in result.get("content", []) if x.get("type") == "text"))
        except (ValueError, TypeError) as exc:
            raise ModelFailure("invalid_model_json") from exc

    async def close(self):
        await self.client.aclose()
