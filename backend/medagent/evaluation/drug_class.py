"""Local evidence for a positive drug-to-family statement in a category question."""
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
import re

from ..understanding.conversation import topic_medications


SENTENCE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
# A quoted positive substring can be inside a question, denial, or hypothetical.
NONASSERTION = re.compile(
    r"[？?]|是否|是不是|会不会|能否|吗|不(?:是|属于|确定|清楚|肯定|知道|敢|好说|一定)|"
    r"无法|不能(?:确定|确认|断定|肯定|说明)|难以|未(?:知|必|能确认|确定)|尚无依据|没有依据|"
    r"没(?:说|有说|把握)|拿不准|说不准|可能|也许|假如|如果|若是|并非|不算|有待核实|"
    r"假设|假定|举例|例如|听说|有人说|请问|请确认|未说明"
)
SUBJECT = r"它|此药|该药|本药|这种药(?:物)?|这款药"
MEMBERSHIP = r"属于|归入|归在|归为|归属(?:于)?|列入|划分?为|算(?:是|作)?|是|为"
FAMILY_SUFFIX = r"(?:这一?[类支族组]|一[类支族组]|家族|类别|系列|类|族|系|组)"


@dataclass(frozen=True)
class DrugClass:
    drug: str
    family: str
    parent: str

    @cached_property
    def assertion(self) -> re.Pattern:
        # Require a direct classification or a short standalone class label. A use,
        # allergy, or comparison that mentions the family does not assert membership.
        qualifier = r"(?:一种|一类|临床上|药理上|分类上|通常|主要|可以|可|被|也|则|的){0,2}"
        parent = rf"(?:(?:{self.parent})(?:中|里|当中)的)?"
        return re.compile(
            rf"(?:^|[，,：:])\s*(?P<subject>{self.drug}|{SUBJECT})?\s*"
            rf"{qualifier}(?:{MEMBERSHIP})?{qualifier}{parent}"
            rf"(?P<family>{self.family}){FAMILY_SUFFIX}"
            rf"[的里中]{{0,3}}(?:一种|一员)?(?:{self.parent}|药物|药)?"
            r"(?=\s*(?:[，,。！；;：:、]|$))"
        )

    @cached_property
    def members_of_family(self) -> re.Pattern:
        # Match the inverse relation only when it names both the family and this drug.
        return re.compile(rf"(?:^|[，,：:])\s*(?P<family>{self.family}){FAMILY_SUFFIX}"
                          rf"(?:{self.parent}|药物)?(?:之中|当中|里面|里|中)?(?:就|也)?"
                          rf"(?:包括|包含|有)(?P<subject>{self.drug})(?=\s*(?:[，,。！；;]|$))")

    @cached_property
    def parent_statement(self) -> re.Pattern:
        return re.compile(rf"(?:^|[，,])\s*(?:{self.drug}|{SUBJECT})\s*(?:{MEMBERSHIP})"
                          rf"(?:在|一种|一类)?(?P<parent>{self.parent})(?:物|中|里|之一)?[。！]?\s*$")

    def supported(self, text: str, quote: str | None = None) -> bool:
        """Anchor the quoted family to an affirmative statement in the same visible field."""
        spans = [(m.start(), m.end()) for m in re.finditer(re.escape(quote), text)] if quote else [(0, len(text))]
        sentences = list(SENTENCE.finditer(text))

        def anchored(sentence: re.Match, offsets: tuple[int, int]) -> bool:
            start, end = offsets
            return any(left <= sentence.start()+start and sentence.start()+end <= right for left, right in spans)

        for index, sentence in enumerate(sentences):
            value = sentence.group()
            if NONASSERTION.search(value):
                continue
            for claim in chain(self.assertion.finditer(value), self.members_of_family.finditer(value)):
                if not claim["subject"] and claim.start() and not re.search(
                    rf"{self.drug}|分类|类别|药物类型|再细分", value[:claim.start()]
                ):
                    continue
                # Explicitly linked neighbouring qualifications can retract the claim.
                before = sentences[index-1].group() if index else ""
                after = sentences[index+1].group() if index+1 < len(sentences) else ""
                if claim["subject"] != self.drug and set(topic_medications(before + value[:claim.start()])) - {self.drug}:
                    continue
                if (re.search(r"(?:不能确认|不确定|不清楚|拿不准|假设)[：:]?。?$", before)
                        or re.match(r"(?:但|不过|可是)?(?:这个(?:说法|分类)|这一点|这点|对此)?我?(?:还|并|也)?"
                                    r"(?:不能确认|不确定|不清楚|拿不准|没有把握)", after)):
                    continue
                if anchored(sentence, claim.span("family")):
                    return True
                # The Judge may quote only the broad class of a two-sentence classification.
                # The adjacent statement must itself classify the drug, not merely describe a use.
                for neighbour in sentences[max(0, index-1):index+2]:
                    parent = self.parent_statement.search(neighbour.group())
                    if (parent and not NONASSERTION.search(neighbour.group()) and not set(topic_medications(neighbour.group())) - {self.drug}
                            and anchored(neighbour, parent.span("parent"))):
                        return True
        return False


DRUG_CLASSES = {
    "amoxicillin-overview": DrugClass("阿莫西林", "青霉素", r"抗生素|抗菌药物?"),
    "metformin-overview": DrugClass("二甲双胍", "双胍", r"降糖药物?"),
}
