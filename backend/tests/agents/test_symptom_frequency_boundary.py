"""How often a symptom happens is not a dose; how often a medicine is taken still is."""
import pytest

from medagent.agents.checks import check_boundaries
from medagent.agents.output_validation import AgentOutputError


@pytest.mark.parametrize("text", ["昨天开始拉肚子，一天四五次。", "腹泻一天四五次，持续约一天。", "夜里咳醒，一天咳十几次。",
                                  "呕吐一天两次。"])
def test_symptom_frequency_is_allowed(text):
    check_boundaries({"text":text})


@pytest.mark.parametrize("text", ["布洛芬一天三次。", "每天吃3次。", "一天两次，每次一片。", "腹泻时止泻药一天两次。"])
def test_medicine_frequency_is_still_blocked(text):
    with pytest.raises(AgentOutputError):
        check_boundaries({"text":text})


@pytest.mark.parametrize("text", ["你从昨天开始腹泻，大约一天四五次，可以在药房买口服补液盐。",
                                  "腹泻一天四五次时，注意补水，口服补液盐请咨询药师。"])
def test_a_medicine_in_another_part_of_the_sentence_does_not_make_the_count_a_dose(text):
    check_boundaries({"text":text})


@pytest.mark.parametrize("text", ["腹泻，一天两次，每次一片。", "拉肚子吃药后，一天两次。"])
def test_a_dose_next_to_a_symptom_is_still_blocked(text):
    with pytest.raises(AgentOutputError):
        check_boundaries({"text":text})


@pytest.mark.parametrize("text", ["免疫力低下（例如正在化疗或患有糖尿病）时应联系医生。", "如果你患有慢性肺病，咳嗽超过3周要联系医生。",
                                  "患有糖尿病等慢性病的人免疫力可能较低。", "假如您患有哮喘，请向医生说明。"])
def test_describing_people_with_a_condition_is_not_a_diagnosis(text):
    check_boundaries({"text":text})


@pytest.mark.parametrize("text", ["你患有支气管炎。", "您可能患有流感。", "这看起来像普通的上呼吸道问题。", "确诊为肺炎。",
                                  "你得的是感冒。", "咳嗽大概率是过敏。"])
def test_telling_the_user_what_they_have_is_still_blocked(text):
    with pytest.raises(AgentOutputError):
        check_boundaries({"text":text})
