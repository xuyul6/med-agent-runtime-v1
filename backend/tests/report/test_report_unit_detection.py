"""A unit the user wrote next to the value counts as provided, even when extraction left the unit field empty."""
import pytest

from medagent.agents.contracts import ReportInput
from medagent.core.contracts import Intent


def gaps(text, **item):
    return ReportInput(text=text, intent=Intent.REPORT, items=[{"name":"中性粒细胞比例", **item}]).information_gaps()


def test_percent_written_after_the_value_is_the_unit():
    text = "血常规里中性粒细胞比例 78%，参考范围 40–75%，是什么意思？"
    assert gaps(text, value="78", reference_range="40–75%") == []


def test_unit_inside_the_extracted_value_is_the_unit():
    assert gaps("中性粒细胞比例78%，参考40-75", value="78%", reference_range="40-75") == []


@pytest.mark.parametrize("text, value, unit", [
    ("尿酸 420 μmol/L，参考范围 150–420", "420", "μmol/L"),
    ("白细胞 11.2×10^9/L，参考 3.5–9.5", "11.2", "×10^9/L"),
    ("血红蛋白 128 g/L，参考 115–150", "128", "g/L"),
    ("某比值 3‰，参考 1–5", "3", "‰"),
])
def test_common_lab_units_next_to_the_value(text, value, unit):
    request = ReportInput(text=text, intent=Intent.REPORT, items=[{"name":"项目", "value":value, "reference_range":"1-2"}])
    assert request.information_gaps() == []
    assert request.items[0].unit == unit


def test_a_bare_number_still_needs_its_unit():
    assert gaps("中性粒细胞比例是78，参考范围40到75", value="78", reference_range="40到75") == ["unit"]


def test_the_unit_must_follow_this_value_not_another_number():
    assert gaps("中性粒细胞比例是78，另一项是5%", value="78", reference_range="40-75") == ["unit"]
    assert gaps("中性粒细胞比例是178%", value="78", reference_range="40-75") == ["unit"]


def test_an_extracted_unit_is_kept_as_given():
    request = ReportInput(text="78%", intent=Intent.REPORT, items=[{"name":"项目", "value":"78", "unit":"百分比", "reference_range":"x"}])
    assert request.items[0].unit == "百分比"
