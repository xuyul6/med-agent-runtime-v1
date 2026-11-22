"""A clinician-facing note can quote the user's account without interpreting it as a lab result."""
import re

from ..core.contracts import Intent, Understanding
from .classification import LEXICON


_READER = r"(?:给|供|交给|带给|拿给)(?:看诊的|接诊的)?(?:医生|医师|大夫|医护)(?:看|参考|阅读)?"
_DOCUMENT = r"(?:就诊|就医|病情)?(?:说明|摘要|记录|小结|清单|描述|文字|话)"
_WRITE = r"(?:整理|写|归纳|汇总|编写|拟)(?:一下)?(?:成|为)?(?:一(?:小?段|份|张|个))?"
# Only an adjacent, affirmative statement of the intended reader completes a writing request.
_PURPOSE = (r"\s*[，,。；;！？!?\n]\s*(?:(?:门诊|就诊|复诊|看诊|看病)(?:那天|时|的时候))?"
            r"(?:我(?:可以|想|打算|好)?)?(?:直接)?")
_ACCOUNT = (r"(?:把|将)(?:我(?:刚才|上面|下面)?(?:说的|提到的|的))?"
            r"(?:这[一二两三四五六七八九十几]+件事|这(?:些|段|几条)|上述|上面|以下|下面|刚才)?的?"
            r"(?:情况|信息|经历|内容|症状和用药|描述)?(?:串起来|合在一起|放在一起)?[，,]?\s*")
_REQUEST = re.compile(
    r"(?:^|(?<=[，,。；;！？!?\n]))\s*(?:我想|我希望|可以|能否|能不能)?(?:请|麻烦)?你?(?:帮我|替我)?"
    rf"(?:{_ACCOUNT})?(?:{_WRITE}{_READER}的?{_DOCUMENT}|{_WRITE}{_DOCUMENT}(?:{_PURPOSE})?{_READER}|{_READER}{_WRITE}{_DOCUMENT})"
    r"(?:可以吗|好吗|吗|吧)?(?=$|[\s，,。；;：:！？!?])[？?]?"
)
_HEALTH_TOPIC = re.compile("|".join(LEXICON[kind] for kind in (Intent.SYMPTOMS, Intent.MEDICINE, Intent.REPORT))
                           + r"|检查|不适|体温|睡眠|过敏", re.I)


def clinician_note(current_request: str, understanding: Understanding) -> str:
    """Separate an explicit writing request from its accompanying verbatim account, in either order."""
    request = _REQUEST.search(current_request)
    if not request:
        return ""
    # Keep whole statements, including time, negation and uncertainty. Do not turn an example,
    # quoted source or separate question into the user's medical history.
    current_quotes = [fact.source.quote for fact in understanding.entities.values()
                      if fact.source.origin == "current_user" and fact.source.turn_id == understanding.turn_id
                      and fact.source.subject == understanding.subject]
    narrative = []
    for part in (current_request[:request.start()], current_request[request.end():]):
        part = part.strip(" \t\n，,。；;：:")
        if not part:
            continue
        if re.search(r"[？?]|如果|假如|假设|举例|例如|比如|示例|原文|摘录|引用|粘贴|[“”「」\"]|"
                     r"请|帮我|只想|只要|不要写|别写|不写进|查一下|怎么|是否|能否|[吗呢](?:$|[，,。])", part):
            continue
        if _HEALTH_TOPIC.search(part) or any(quote in part for quote in current_quotes):
            narrative.append(part)
    return "【给医生的说明（本轮自述）】\n"+"。".join(narrative)+"。" if narrative else ""
