"""Common drug-name lexicon for literal recognition; names outside it are left to grounded model proposals."""
import json
import re
from ..core.config import ROOT

LEXICON_PATH = ROOT/"resources"/"medication-names.json"


def load(path=LEXICON_PATH) -> dict[str, str]:
    """Written form (casefolded) -> canonical generic name."""
    body = json.loads(path.read_text())
    canonical = {}
    for item in body["medications"]:
        for written in [item["name"], *item["aliases"]]:
            key = written.casefold()
            if canonical.setdefault(key, item["name"]) != item["name"]:
                raise ValueError("ambiguous_medication_alias:"+written)
    return canonical


CANONICAL = load()


# "艾司奥美拉唑" (esomeprazole) or "复方…" products are different drugs from the name they contain.
PREFIX_GUARD = r"(?<!艾司)(?<!左)(?<!右)(?<!复方)"


def _alternation(forms) -> str:
    # Longest first so "地氯雷他定" is not read as "氯雷他定"; ASCII aliases need word boundaries.
    parts = []
    for form in sorted(forms, key=len, reverse=True):
        escaped = re.escape(form).replace(r"\ ", r"\s?")
        parts.append(r"(?<![a-z])"+escaped+r"(?![a-z])" if form.isascii() else PREFIX_GUARD+escaped)
    return "|".join(parts)


# "胰岛素抵抗" or "叶酸含量" name a physiological concept or nutrient level, not a drug being asked about.
NOT_A_DRUG_AFTER = r"抵抗|敏感|分泌|水平|含量|浓度|受体|缺乏|检测|指数|瘤"
NAMES = "(?:" + _alternation(CANONICAL) + ")(?!" + NOT_A_DRUG_AFTER + ")"
NAME_PATTERN = re.compile(NAMES, re.I)


def canonical(value: str | None) -> str | None:
    if value is None:
        return None
    return CANONICAL.get(re.sub(r"\s+", " ", value.strip()).casefold(), value.strip())
