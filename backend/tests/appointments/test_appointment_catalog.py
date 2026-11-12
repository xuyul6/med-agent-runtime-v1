from datetime import date

from medagent.appointments.catalog import HospitalCatalog
from medagent.appointments.params import abandons, appointment_action, asks_question, parse_day, parse_window
from medagent.core.config import ROOT

TODAY = date(2026, 11, 3)


def test_parse_day():
    assert parse_day("明天", TODAY) == ("valid", date(2026, 11, 4))
    assert parse_day("那就明天上午吧", TODAY) == ("valid", date(2026, 11, 4))
    assert parse_day("周五", TODAY).value == date(2026, 11, 6)
    assert parse_day("周二", TODAY).value == date(2026, 11, 10)
    assert parse_day("下周一", TODAY).value == date(2026, 11, 9)
    assert parse_day("11月2日", TODAY).value == date(2027, 11, 2)
    assert parse_day("2026-11-05", TODAY).value == date(2026, 11, 5)
    assert parse_day("11月31日", TODAY).state == "invalid"
    assert parse_day("明天或者后天", TODAY).state == "invalid"
    assert parse_day("改天", TODAY).state == "absent" and parse_day(None, TODAY).state == "absent"


def test_parse_window():
    assert parse_window("明天上午") == ("valid", "am") and parse_window("下午三点") == ("valid", "pm")
    assert parse_window("晚上").state == "invalid" and parse_window("上午下午都行").state == "invalid"
    assert parse_window("明天").state == "absent"


def test_appointment_action():
    assert appointment_action("帮我挂明天神经内科的号") == "create"
    assert appointment_action("帮我取消明天的预约") == "cancel"
    assert appointment_action("能帮我挂明天上午神经内科的号吗") == "create"
    assert appointment_action("帮我挂11月8日下午神经内科的号") == "create"
    for quote in ["确认", "好的", "怎么预约神经内科", "不挂号了", "算了不约了", "取消吧", None]:
        assert appointment_action(quote) is None


def test_followup_wording():
    assert abandons("不挂了，神经内科一般看什么？") and abandons("算了") and not abandons("神经内科")
    assert asks_question("神经内科一般看什么") and not asks_question("明天上午")


def test_catalog_matching_and_slots():
    catalog = HospitalCatalog.load(ROOT/"resources"/"mock-hospital.json")
    assert [d.id for _, d in catalog.match_department("挂神经科吧")] == ["neurology"]
    assert len(catalog.match_department("皮肤科")) == 2
    assert [h.id for h, _ in catalog.match_department("皮肤科", "demo-community")] == ["demo-community"]
    assert [h.id for h in catalog.match_hospital("社区医院")] == ["demo-community"]
    assert catalog.match_department("骨科") == []
    assert not catalog.bookable(TODAY, TODAY) and catalog.bookable(date(2026, 11, 17), TODAY) and not catalog.bookable(date(2026, 11, 18), TODAY)
    am = catalog.slots("demo-general", "neurology", date(2026, 11, 4), "am", TODAY)
    assert [s.slot_id for s in am] == ["demo-general:neurology:2026-11-04:am"] and am[0].capacity == 3
    assert catalog.slots("demo-general", "neurology", date(2026, 11, 8), "pm", TODAY) == []
    assert catalog.slot("demo-general:neurology:2026-11-04:am").department == "神经内科"


import pytest


@pytest.mark.parametrize("quote, action", [
    ("帮我挂神经内科", "create"), ("我要挂个神经内科明天上午", "create"), ("挂明天上午的神经内科", "create"),
    ("约个明天的神经内科", "create"), ("帮我把明天的号取消", "cancel"), ("取消明天神经内科", "cancel"),
    ("把明天神经内科的预约退了", "cancel"), ("明天神经内科的预约不要了", "cancel"),
    ("确认预约", None), ("好的，预约吧", None), ("确认", None)])
def test_action_wording(quote, action):
    assert appointment_action(quote) == action


def test_replies_that_answer_the_followup():
    assert not abandons("不约明天了，改后天下午") and abandons("不约了") and abandons("算了，不挂了")
    assert asks_question("神经内科一般看什么") and asks_question("明天上午可以吗")
