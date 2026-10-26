import re
import time
from .contracts import Risk

SIGNS = {"breathing": r"呼吸困难|喘不过气|无法呼吸|嘴唇发紫",
         "chest": r"胸痛|胸口剧痛|胸口压榨",
         "consciousness": r"意识不清|叫不醒|昏迷|失去意识",
         "bleeding": r"大量出血|呕血|咳血不止",
         "neurological": r"突发.{0,5}(?:口齿不清|一侧无力)|突然.{0,5}(?:口齿不清|一侧无力)"}


class RiskGate:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.hints = {}

    def screen(self, text: str, scope: str) -> Risk:
        now = self.clock()
        self.hints = {k: v for k, v in self.hints.items() if v > now}
        educational = bool(re.match(r"\s*(?:请)?(?:科普|介绍一下|解释一下|什么是)", text)) and not re.search(r"我现在|现在.{0,5}(?:胸痛|呼吸困难)|此刻", text)
        findings = []
        for clause in re.split(r"[，。！？；\n]|但是|不过(?=我|现在|目前)|但", text):
            for code, pattern in SIGNS.items():
                for match in re.finditer(pattern, clause):
                    prefix = clause[max(0, match.start() - 8):match.start()]
                    denied = re.search(r"(?:没有|无|否认|不伴|并非|不是|未出现).{0,4}$", prefix)
                    historical = re.search(r"去年|以前|曾经|上个月|已经缓解|已恢复", clause) and not re.search(r"现在|目前|此刻|再次", clause)
                    if not denied and not historical and not educational:
                        findings.append(code)
        if findings:
            self.hints[scope] = now + 300
            return Risk(level="CRITICAL", reasons=sorted(set(findings)))
        if re.search(r"已就医|已经到医院|已经缓解|已恢复|症状消失", text):
            self.hints.pop(scope, None)
        if scope in self.hints and re.search(r"更严重|加重|还是这样|越来越|又发作", text):
            return Risk(level="CRITICAL", reasons=["recent_urgent_followup"])
        if re.search(r"怀孕|孕妇|婴儿|儿童|孩子|过量|多吃了|相互作用|一起吃", text):
            return Risk(level="CAUTION", reasons=["professional_review_needed"])
        return Risk()
