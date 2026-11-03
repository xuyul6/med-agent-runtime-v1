"""Parse appointment wording. A value the user gave but that cannot be used is invalid, never treated as absent."""
import re
from datetime import date, timedelta
from typing import Literal, NamedTuple

WINDOW_NAMES = {"am": "上午", "pm": "下午"}
WEEKDAYS = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
RELATIVE = {"今天": 0, "明天": 1, "后天": 2, "大后天": 3}
DAY_PATTERN = re.compile(r"(?P<iso>\d{4}-\d{1,2}-\d{1,2})|(?P<month>\d{1,2})月(?P<day>\d{1,2})[日号]"
                         r"|(?P<next>下)?(?:周|星期|礼拜)(?P<weekday>[一二三四五六日天])|(?P<relative>大后天|今天|明天|后天)")
AM, PM, UNSUPPORTED_WINDOW = r"上午|早上|早晨", r"下午|午后", r"晚上|夜里|夜间|中午|凌晨"
HOW_TO = r"怎么|如何|流程|步骤|方式"
NEGATED = r"不(?:用|要|想|需要)?(?:挂|约|预约)|算了"
CANCEL = r"(?:取消|退掉?)[^，。,.]{0,8}(?:预约|号|门诊)|退号"
# A date, a time of day and a department can sit between 挂 and 号: 挂11月8日下午神经内科的号.
CREATE = r"挂号|挂[^，。,.]{0,16}号|预约|约.{0,6}(?:号|门诊|医生|科)"
ABANDON = r"不(?:用|要|想)?(?:挂|约|预约)了?|算了|先不(?:挂|约)?了?|不用了|不需要了|取消吧"
QUESTION = r"[?？]|吗|什么|怎么|如何|为什么|哪些|能不能|是否|多少"


class Parsed(NamedTuple):
    state: Literal["absent", "valid", "invalid"]
    value: date | str | None


def _day(match: re.Match, today: date) -> date | None:
    try:
        if match["iso"]:
            return date.fromisoformat("-".join(part.zfill(2) for part in match["iso"].split("-")))
        if match["month"]:
            day = date(today.year, int(match["month"]), int(match["day"]))
            return day if day >= today else date(today.year + 1, day.month, day.day)
    except ValueError:
        return None
    if match["relative"]:
        return today + timedelta(days=RELATIVE[match["relative"]])
    target = WEEKDAYS[match["weekday"]]
    if match["next"]:
        return today - timedelta(days=today.weekday()) + timedelta(days=7 + target)
    return next(today + timedelta(days=offset) for offset in range(1, 8) if (today + timedelta(days=offset)).weekday() == target)


def parse_day(text: str | None, today: date) -> Parsed:
    matches = list(DAY_PATTERN.finditer(text or ""))
    if not matches:
        return Parsed("absent", None)
    days = {_day(match, today) for match in matches}
    if None in days or len(days) > 1:
        return Parsed("invalid", "、".join(match[0] for match in matches))
    return Parsed("valid", days.pop())


def parse_window(text: str | None) -> Parsed:
    text = text or ""
    if found := re.search(UNSUPPORTED_WINDOW, text):
        return Parsed("invalid", found[0])
    am, pm = re.search(AM, text), re.search(PM, text)
    if am and pm:
        return Parsed("invalid", am[0] + pm[0])
    if am or pm:
        return Parsed("valid", "am" if am else "pm")
    return Parsed("absent", None)


def appointment_action(quote: str | None) -> Literal["create", "cancel"] | None:
    if not quote or re.search(HOW_TO, quote) or re.search(NEGATED, quote):
        return None
    if re.search(CANCEL, quote):
        return "cancel"
    return "create" if re.search(CREATE, quote) else None


def abandons(text: str) -> bool:
    return bool(re.search(ABANDON, text or ""))


def asks_question(text: str) -> bool:
    return bool(re.search(QUESTION, text or ""))
