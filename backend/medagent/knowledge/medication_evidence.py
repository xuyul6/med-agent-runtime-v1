"""Conservative document applicability, independent of model tool arguments."""
import re
from ..core.contracts import Reference
from ..agents.output_validation import AgentOutputError
from ..understanding.medications import canonical


IDENTITY_FIELDS = ("formulation", "population", "product_name", "strength", "region")
UNKNOWN_MARKERS = frozenset({"未知", "不详", "未提供", "未明确", "不清楚", "unknown", "unspecified"})


def query_identity(request) -> dict:
    return {"name":getattr(request,"medication_name",None),
            **{field:getattr(request,field,None) for field in IDENTITY_FIELDS}}


def identity_parts(value: str) -> list[str]:
    # Accept only whole aliases, including a bilingual parenthetical rendering; never a substring match.
    value = value.strip().replace("（","(").replace("）",")")
    match = re.fullmatch(r"([^()]+)\(([^()]+)\)",value)
    return [part.strip() for part in match.groups()] if match else [value]


def normalized(value: str | None, field: str) -> str:
    value = re.sub(r"\s+", "", (value or "").casefold())
    if field == "strength": value = value.replace("毫克", "mg")
    aliases = {
        "region": {"美国":"US", "us":"US", "usa":"US", "英国":"UK", "uk":"UK", "中国":"CN", "中国大陆":"CN", "cn":"CN"},
        "population": {"成人":"adults_18_plus", "成年人":"adults_18_plus", "18岁及以上":"adults_18_plus", "adult":"adults_18_plus"},
        "formulation": {"口服包衣片":"coated_tablet_oral", "包衣片":"coated_tablet_oral", "coatedtablet":"coated_tablet_oral"},
    }
    return aliases.get(field, {}).get(value, value)


def rejection_reason(ref: Reference, request) -> str | None:
    target = getattr(request, "medication_name", None)
    if target:
        matches = canonical(target).casefold() in {alias.casefold() for alias in ref.entities}
    else:
        matches = any(re.search(r"(?<![a-z])"+re.escape(alias)+r"(?![a-z])", request.text, re.I)
                      if alias.isascii() else alias in request.text for alias in ref.entities)
    if not matches:
        return "medication_entity_mismatch"
    scope = ref.medication_scope
    if not scope or not ref.provenance:
        return "unverified_source_scope"
    is_label = ref.provenance.document_type == "product_label"
    if getattr(request, "query_type", "overview") == "product_information" and not is_label:
        return "overview_is_not_product_label"
    fields = ("region", "population", "product_name", "formulation", "strength") if is_label else ("region", "population")
    for field in fields:
        supplied = getattr(request, field, None)
        if not supplied:
            if is_label: return "missing_"+field
            continue
        if normalized(supplied, field) != normalized(getattr(scope, field), field):
            return field+"_mismatch"
    return None


def eligible_references(records, request):
    return [ref for ref in records if ref.approved and rejection_reason(ref, request) is None]


def validate_output_identity(data: dict, request, references=()) -> list[str]:
    payload = data.get("payload", {})
    if not isinstance(payload,dict) or not isinstance(payload.get("medication_identity",{}),dict):
        raise AgentOutputError("invalid_output_shape",["medication_identity"])
    identity = payload.get("medication_identity", {})
    if identity.get("name") is not None and not isinstance(identity["name"],str):
        raise AgentOutputError("invalid_output_shape",["name"])
    changed = []
    for field in IDENTITY_FIELDS:
        value = identity.get(field)
        supplied = getattr(request, field, None)
        if supplied is None and isinstance(value,str) and value.strip().casefold() in UNKNOWN_MARKERS:
            identity[field] = None
            changed.append(field)
        elif value is not None:
            if not isinstance(value,str) or not supplied or not all(normalized(part,field) == normalized(supplied,field) for part in identity_parts(value)):
                raise AgentOutputError("unconfirmed_medication_identity",[field])
            if value != supplied:
                identity[field] = supplied
                changed.append(field)
        elif supplied is not None:
            # Trusted input metadata can be copied by code; source fields cannot fill unknowns.
            identity[field] = supplied
            changed.append(field)
    target = getattr(request, "medication_name", None)
    parts = identity_parts(identity["name"]) if isinstance(identity.get("name"),str) else []
    aliases = [{alias.casefold() for alias in ref.entities} for ref in references]
    if identity.get("name") and not target and not any(
        {part.casefold() for part in parts} <= names for names in aliases
    ):
        raise AgentOutputError("unconfirmed_medication_name",["name"])
    if target and identity.get("name") and identity["name"].casefold() != target.casefold():
        if not parts or not any({target.casefold(), *[part.casefold() for part in parts]} <= names for names in aliases):
            raise AgentOutputError("medication_identity_target_mismatch",["name"])
        identity["name"] = target
        changed.append("name")
    return changed
