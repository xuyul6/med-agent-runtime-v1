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
# The booking may come before or after the verb: 取消明天的预约 / 把明天的号退了 / 明天的预约不要了.
CANCEL = (r"(?:取消|退掉?)[^，。,.]{0,12}(?:预约|号|门诊|科)|(?:预约|号|门诊)[^，。,.]{0,12}(?:取消|退了|退掉|不要了)"
          r"|把[^，。,.]{0,12}(?:预约|号)(?:退|取消)|退号")
# A date, a time of day and a department can sit between 挂 and 号, and the 号 may be left out: 挂明天上午的神经内科.
# Several slots (11月5日上午和11月6日上午) make the span longer; 大约 is not a booking.
CREATE = r"挂号|挂(?:个|一个)?[^，。,.]{0,30}(?:号|科|门诊)|预约|(?<!大)约[^，。,.]{0,24}(?:号|门诊|医生|科)"
# Agreement in chat after a card is not a request; only the button confirms.
CONFIRM_ONLY = r"^(?:确认|好的?|可以|行)[，,]?(?:预约|挂号)?吧?[。！!]?$"
# 不约明天了，改后天下午 replaces a condition; it does not give up the booking.
REPLACEMENT = r"改(?:约|成|到)?|换(?:成|到)?"
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


def parse_items(quote: str | None) -> list[tuple[str, str | None]]:
    """Two or more slots in one verified quote: each date with the time of day in its own segment.

    Only the last segment naming a time of day, as in 11月5日和11月6日下午, means both. One slot returns [] and keeps the single path.
    """
    matches = list(DAY_PATTERN.finditer(quote or ""))
    windows = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(quote)
        found = re.search(AM + "|" + PM + "|" + UNSUPPORTED_WINDOW, quote[match.end():end])
        windows.append(found[0] if found else None)
    if len(windows) > 1 and windows[-1] and not any(windows[:-1]):
        windows = [windows[-1]] * len(windows)
    items = list(dict.fromkeys(zip((match[0] for match in matches), windows)))
    return items if len(items) > 1 else []


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
    if not quote or re.search(HOW_TO, quote) or re.fullmatch(CONFIRM_ONLY, quote.strip()):
        return None
    if re.search(NEGATED, quote) and not re.search(CANCEL, quote):
        return None
    if re.search(CANCEL, quote):
        return "cancel"
    return "create" if re.search(CREATE, quote) else None


def abandons(text: str) -> bool:
    return bool(re.search(ABANDON, text or "")) and not re.search(REPLACEMENT, text or "")


def asks_question(text: str) -> bool:
    return bool(re.search(QUESTION, text or ""))
