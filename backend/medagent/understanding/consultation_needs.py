"""Bounded recognition of one ambiguous fever question and its follow-up."""
import math
import re


INITIAL = re.compile(r"^(?:我|本人)?(?:发烧|发热|低烧|低热)(?:时|了|的时候)?(?:应该|可以|要|该)?(?:吃什么|吃点什么)[？?。！!]*$")
FOLLOWUP = re.compile(r"^(?:我|是|大概|约|从|已经|持续|开始|到现在|一直|有|昨天|昨晚|今天|今晚|前天|早上|晚上|上午|下午|夜里|以来|了|的|左右|低烧|低热|发烧|发热|小时|个小时|天|夜|半|一|二|三|四|五|六|七|八|九|十|两|\d|\s|[，,。！!])+$")
FOOD = re.compile(r"^(?:我|是|想|要|想问|想了解|问的是|说的是|了解|饮食|食物|吃饭|吃的|饭菜|吃|饭|方面|什么|呢|呀|[\s，,。！？?])+$")
MEDICINE = re.compile(r"^(?:我|是|想|要|想问|想了解|问的是|说的是|了解|用药|药物|药品|退烧药|药|吃|方面|什么|呢|呀|[\s，,。！？?])+$")


def is_fever_followup(text: str) -> bool:
    value = text.strip()
    return bool(len(value) <= 60 and FOLLOWUP.fullmatch(value) and re.search(r"(?:发烧|发热|低烧|低热)", value))


def consultation_need(text: str, state: dict, *, subject: str, subject_changed: bool,
                      now: float, recent_critical: bool) -> tuple[str | None, dict | None]:
    """Return a finite action only; never infer a patient fact or store raw text."""
    if recent_critical:
        return None, None
    if INITIAL.fullmatch(text.strip()):
        return "clarify", None
    if subject_changed or not isinstance(state, dict):
        return None, None
    pending = state.get("pending_consultation")
    if not isinstance(pending, dict) or pending.get("kind") != "food_or_medicine" or pending.get("subject") != subject:
        return None, None
    created = pending.get("created_at")
    if type(now) not in (int, float) or not math.isfinite(now) or type(created) not in (int, float) or not math.isfinite(created) or not 0 <= now-created <= 300:
        return None, None
    if type(pending.get("continuations")) is not int or not 0 <= pending["continuations"] <= 2:
        return None, None
    if not isinstance(pending.get("source_turn_id"), str) or not pending["source_turn_id"]:
        return None, None
    messages = state.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None, None
    user, assistant = messages[-2:]
    last_id = pending.get("last_turn_id")
    if (not isinstance(user, dict) or not isinstance(assistant, dict) or not isinstance(last_id, str) or not last_id
        or user.get("role") != "user" or assistant.get("role") != "assistant"
        or user.get("turn_id") != last_id or assistant.get("turn_id") != last_id
        or state.get("last_intent_turn_id") != last_id
        or user.get("turn_index") != state.get("turn_counter") or assistant.get("turn_index") != state.get("turn_counter")
        or "consultation_focus" not in state.get("last_missing_fields", [])):
        return None, None
    value = text.strip()
    if FOOD.fullmatch(value) and re.search(r"饮食|食物|吃的|饭", value):
        return "food", pending
    if MEDICINE.fullmatch(value) and re.search(r"药", value):
        return "medicine", pending
    if pending["continuations"] < 2 and is_fever_followup(value):
        return "clarify_followup", pending
    return None, None
