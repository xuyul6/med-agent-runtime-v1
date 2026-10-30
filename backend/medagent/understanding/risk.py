"""Deterministic emergency screening; an engineering filter that cannot recognise every emergency.

The rules run first and work without a model. A model signal may raise a turn to critical afterwards but never lowers one.
"""
import re
import time
from ..core.contracts import Risk

RULE_VERSION = "screening-3"
# An unresolved urgent sign keeps short follow-ups ("还是这样") urgent for this long.
HINT_SECONDS = 1800
PAIN = "(?:痛|疼)"
# Words that sit inside a sign ("呼吸很困难"); the matched sign is still subject to the denial checks.
DEGREE = r"(?:很|好|太|挺|特别|非常|十分|极其|越来越|有点|有些|都|也|已经|更|这么|那么){0,2}"
SUDDEN = r"(?:突然|突发|忽然|一下子)"
WEAK = r"(?:动不了|没(?:有)?(?:力|知觉|感觉)|无力|使不上劲|麻|抬不起)"
# Sides need 边/侧 for 一 and 半 so that "一手" and "半手" stay ordinary words.
SIDE_LIMB = r"(?:[左右](?:边|侧)?|[一半](?:边|侧))的?(?:手(?!机)|胳膊|手臂|腿|脚)"
SIDE_BODY = r"(?:[左右一半])(?:边|侧)的?(?:手脚|胳膊和?腿|手和脚|手臂和腿|身子|身体|肢体)"

SIGNS = {
    "breathing": "|".join([
        r"呼吸" + DEGREE + r"困[难男南]", r"喘不[过上]气", r"上不来气", r"(?:无法|不能)呼吸", r"吸不进(?:去)?(?:气)?", r"一口气都吸不",
        r"憋得(?:慌|喘|说不出)", r"憋[紫青]", r"窒息", r"(?:嘴唇|口唇)" + DEGREE + r"发?[紫青]", r"喘得(?:说不出话|很厉害|厉害|不行)", r"坐着都喘",
        r"呼吸(?:很|特别|非常)?急促?(?:很急)?", r"肋骨.{0,4}凹", r"卡住(?:了)?(?:喉咙|嗓子)", r"哮喘发作",
        r"can'?t ?breathe", r"cannot ?breathe", r"shortness ?of ?breath"]),
    "chest": "|".join([
        r"胸[口部]?" + DEGREE + r"(?:剧|闷|绞|压榨)?" + PAIN, r"胸口(?:剧痛|压榨|像压|.{0,4}压了)", r"胸[口部]?闷" + PAIN,
        r"心口" + DEGREE + PAIN, r"心脏.{0,3}(?:绞痛|剧痛|" + PAIN + ")", r"chest ?pain"]),
    "consciousness": "|".join([
        r"意识(?:不清|模糊|丧失)", r"神志不清", r"叫不(?:太)?醒", r"(?:拍|喊|叫|摇).{0,4}(?:不醒|没(?:有)?反应)", r"昏迷",
        r"昏(?:过去|倒)", r"晕(?:倒|过去)", r"失去意识", r"不省人事", r"倒地", r"说胡话", r"认不出.{0,4}人", r"反应(?:很|特别)?迟钝"]),
    "bleeding": "|".join([
        r"大量出血", r"大出血", r"出血不止", r"血.{0,12}止不住", r"止不住.{0,3}血", r"喷血", r"血.{0,2}一直.{0,2}(?:流|冒)",
        r"一直(?:在)?(?:流|冒)血", r"血.{0,2}往外冒", r"呕血", r"吐(?:了)?(?:好多|很多|大量|一口)?血", r"咳血不止", r"大口咳血",
        r"黑便", r"柏油(?:样)?(?:的)?便"]),
    "neurological": "|".join([
        SUDDEN + r".{0,6}(?:口齿不清|说话.{0,2}不清|说不(?:清|出)|含糊不清|一侧无力|半边|嘴歪|重影|走路不稳|站不稳|看不见|剧烈头" + PAIN + ")",
        r"嘴(?:巴|角)?" + DEGREE + r"歪了", r"口角歪斜", r"眼歪嘴斜", SIDE_BODY + r".{0,6}" + WEAK, r"一侧(?:肢体)?无力", r"(?:一边|半边)脸.{0,2}麻",
        SIDE_LIMB + r".{0,3}" + SUDDEN + r".{0,3}" + WEAK, SUDDEN + r".{0,3}" + SIDE_LIMB + r".{0,6}" + WEAK,
        r"手.{0,2}抬不起来", r"剧烈的?头" + PAIN, r"头" + PAIN + r".{0,6}(?:炸开|欲裂|裂开)"]),
    "anaphylaxis": "|".join([
        r"(?:喉咙|嗓子)(?:发紧|肿)", r"(?:舌头|嘴唇).{0,2}肿", r"过敏性休克",
        r"(?:吃了|被).{0,8}(?:后|了).{0,6}脸.{0,2}肿", r"全身.{0,4}(?:疹|风团).{0,8}(?:喘|紧|呼吸)"]),
    "self_harm": "|".join([
        r"自杀", r"轻生", r"不想活", r"活着没(?:意思|意义)", r"结束(?:这一切|生命|自己)", r"割腕", r"跳楼", r"跳下去", r"寻死",
        r"伤害自己"]),
    "poisoning": "|".join([
        r"误(?:喝|服|吞|食)", r"吞了.{0,4}(?:电池|磁铁)", r"农药", r"(?:一整|整|半)瓶", r"(?:吃|服|吞)了?.{0,3}(?:二十|十几|几十|[1-9]\d)片",
        r"一下子吃了", r"烧炭", r"煤气中毒", r"一氧化碳", r"中毒"]),
    "seizure": "|".join([r"抽搐", r"(?:高烧|发烧|全身|孩子|宝宝).{0,4}抽筋", r"抽风", r"全身抽动", r"口吐白沫", r"眼睛.{0,2}往上翻", r"癫痫发作"]),
    "trauma": "|".join([
        r"头.{0,4}(?:流血|出血)", r"车祸.{0,8}(?:流血|出血|昏|动不了|骨折)", r"(?:摔|跌).{0,10}(?:变形|动不了|骨折)",
        r"(?:腿|胳膊|脚|手臂|手腕|骨头)(?:都|也|好像)?变形", r"大面积(?:烫伤|烧伤)", r"(?:烫|烧)伤.{0,6}(?:脱落|大面积)", r"皮.{0,2}(?:都)?脱落"]),
    "abdominal": "|".join([r"肚子.{0,3}(?:剧痛|" + PAIN + r"得.{0,2}(?:打滚|受不了))", r"腹部剧痛", r"腹痛.{0,4}剧烈", r"硬得像木板", r"板状腹"]),
    "heat": r"中暑.{0,10}(?:神志|昏|不出汗|抽)|热射病",
    "generic_distress": r"救命|救救我|(?:快|要)不行了|求救",
}

PERSONAL = r"我(?:现在|刚才|正在|也)|我[爸妈家]|我的(?:孩子|家人)"
PRESENT = r"我现在|现在|目前|此刻|刚才|刚刚|正在|一直|突然|我[爸妈]|爷爷|奶奶|老公|老婆|孩子|宝宝|他|她"
# Questions about a sign rather than reports of one.
EDUCATIONAL = (r"科普|介绍一下|解释一下|什么是|了解一下|想了解|想学习|学习一下|在学习|论文|课上|培训|护理记录|有什么区别|"
               r"常见原因|有哪些(?:原因|特点|症状)|有什么特点|为什么会")
# Only explicit media words: exclusions suppress real emergencies when they misfire (e.g. "怎么拍都不醒").
FICTION = r"电视剧|电影|小说|剧里|剧情|游戏|动画|漫画|角色|主角"
HYPOTHETICAL = r"如果|假如|万一|要是|假设|一旦|以防"
FIGURATIVE = r"压力|累得|累到|忙得|作业|加班|笑(?:得|到)"
DENIED_BEFORE = r"(?:没有|没|无|否认|不伴|并非|不是|未出现)[^，。！？；]{0,4}$"
DENIED_AFTER = r"^[？?，,：:]*(?:目前|现在)?(?:都|也)?(?:没有|没|无)(?!.{0,2}(?:缓解|好转|好|停|减轻))"
HISTORICAL = r"去年|以前|曾经|上个月|前几年|之前|做过|已经缓解|已恢复|治好了"
CURRENT = r"(?:现在|目前|此刻|再次|又)(?!.{0,3}(?:好了|正常|没事|恢复))"
CLAUSES = r"[，,。！!？?；;\n]|但是|不过(?=我|现在|目前)|但"
FOLLOWUP = r"更严重|加重|还是这样|越来越|又发作|比刚才|比之前|更厉害|还在" + PAIN + r"|没有?好转|没缓解|更难受|更" + PAIN
RESOLVED = r"已就医|已经到医院|到医院了|已经缓解|已恢复|症状消失"
CAUTION = (r"怀孕|孕妇|备孕|哺乳|婴儿|新生儿|宝宝|幼儿|儿童|孩子|小朋友|过量|多吃了|吃多了|吃重了|又吃了一次|重复吃|相互作用|一起吃|同时吃|"
           r"再加一种|肝功能|肾功能|肝肾|透析|华法林|抗凝")


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower().replace("’", "'")


def grounded_quote(quote, text: str) -> str:
    """A model signal counts only as a quote of the current turn; injected text can at most cause a false alarm or an extra role."""
    if not isinstance(quote, str) or len(normalize(quote)) < 2 or normalize(quote) not in normalize(text):
        return ""
    return quote


def context_excludes(text: str) -> bool:
    """Whole-message framing that turns a sign word into a question, story or hypothesis."""
    if re.search(FICTION, text):
        return not re.search(PERSONAL, text)
    if re.search(PRESENT, text):
        return False
    return any(re.search(pattern, text) for pattern in (EDUCATIONAL, HYPOTHETICAL))


class RiskGate:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.hints = {}

    def findings(self, text: str) -> list[str]:
        text = normalize(text)
        if context_excludes(text):
            return []
        found, offset = [], 0
        for clause in re.split(CLAUSES, text):
            start = text.find(clause, offset)
            offset = start + len(clause)
            historical = re.search(HISTORICAL, clause) and not re.search(CURRENT, clause)
            figurative = re.search(FIGURATIVE, clause) and not re.search(r"突然|现在|一直", clause)
            for code, pattern in SIGNS.items():
                for match in re.finditer(pattern, clause):
                    before = clause[max(0, match.start()-8):match.start()]
                    after = text[start+match.end():start+match.end()+12]
                    if re.search(DENIED_BEFORE, before) or re.match(DENIED_AFTER, after):
                        continue
                    if historical or figurative and code == "breathing" or re.match(r"中心", after):
                        continue
                    if re.search(r"(?:治疗|缓解)$", before) or code == "poisoning" and re.search(r"(?:轻微|有点|一点)的?(?:食物)?$", before):
                        continue
                    found.append(code)
        return sorted(set(found))

    def escalate(self, scope: str) -> Risk:
        """Raise a turn to critical on a grounded model signal; follow-ups stay urgent as after a rule hit."""
        self.hints[scope] = self.clock() + HINT_SECONDS
        return Risk(level="CRITICAL", reasons=["model_urgent_signal"], rule_version=RULE_VERSION)

    def screen(self, text: str, scope: str) -> Risk:
        now = self.clock()
        self.hints = {k: v for k, v in self.hints.items() if v > now}
        if found := self.findings(text):
            self.hints[scope] = now + HINT_SECONDS
            return Risk(level="CRITICAL", reasons=found, rule_version=RULE_VERSION)
        normalized = normalize(text)
        if re.search(RESOLVED, normalized):
            self.hints.pop(scope, None)
        elif scope in self.hints and re.search(FOLLOWUP, normalized):
            return Risk(level="CRITICAL", reasons=["recent_urgent_followup"], rule_version=RULE_VERSION)
        if re.search(CAUTION, normalized):
            return Risk(level="CAUTION", reasons=["professional_review_needed"], rule_version=RULE_VERSION)
        return Risk(rule_version=RULE_VERSION)
