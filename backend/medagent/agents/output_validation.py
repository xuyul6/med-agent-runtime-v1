"""Bounded diagnostic codes: never serialize model values or exception messages."""
from pydantic import ValidationError


ERRORS = {
    "invalid_output_json": ("json", True),
    "invalid_output_shape": ("schema", False),
    "invalid_model_envelope": ("envelope", False),
    # Booking role: the draft must be the one checked candidate; neither is repairable by the model.
    "appointment_draft_missing": ("appointment", False),
    "appointment_draft_mismatch": ("appointment", False),
    "unconfirmed_medication_identity": ("identity", True),
    "unconfirmed_medication_name": ("identity", True),
    "medication_identity_target_mismatch": ("identity", True),
    "unsupported_reference": ("evidence", False),
    "payload_references_not_in_answer": ("evidence", False),
    "reference_outside_role_scope": ("evidence", False),
    "report_range_claim_without_exact_source": ("evidence", False),
    "institution_source_not_supported": ("evidence", False),
    "requested_evidence_not_returned": ("evidence", False),
    "medical_boundary_violation": ("boundary", False),
    "individual_dosing_not_supported": ("boundary", False),
    "unexpected_tools_during_repair": ("protocol", False),
}
SAFE_FIELDS = frozenset({"role", "status", "answer", "missing_fields", "payload", "text", "questions", "next_steps",
    "limitations", "reference_keys", "medication_identity", "name", "medication_name", "formulation", "population",
    "product_name", "strength", "region", "source_scope", "evidence_items", "unresolved_questions", "task_summary",
    "suggested_role", "symptom_summary", "timeline", "care_information", "report_items", "value", "unit",
    "reference_range", "term_explanations", "questions_for_clinician", "preparation_checklist", "institution_facts",
    "needs_verification", "risk_notice", "known_facts", "handoff_connected", "unknown_field"})
ROOT_RULE_FIELDS = {
    "clarification_requires_missing_fields":("status","missing_fields"),
    "missing_fields_require_clarification":("status","missing_fields"),
    "task_summary_required":("task_summary",),
    "symptom_summary_required":("symptom_summary",),
    "medication_evidence_required":("medication_identity","source_scope","evidence_items"),
    "report_information_required":("report_items","term_explanations"),
    "visit_checklist_required":("preparation_checklist",),
}
FIELD_RULES = {
    "missing":"required_field", "extra_forbidden":"unexpected_field",
    **{name:"wrong_field_type" for name in ("string_type","list_type","dict_type","model_type","model_attributes_type")},
    **{name:"invalid_field_value" for name in ("literal_error","enum","too_short","too_long","string_too_short","string_too_long")},
}
BOUNDARY_RULES = ("dose_amount", "dose_frequency", "dose_relative", "medication_decision", "safety_guarantee",
                  "diagnosis", "prescription", "action_claim")
SAFE_RULES = frozenset([*ROOT_RULE_FIELDS,*FIELD_RULES.values(),*BOUNDARY_RULES,"schema_validation_failed","unknown_rule"])
REPORT_EVIDENCE_REASONS = frozenset({"no_sources_cited", "exact_sentence_in_uncited_source",
                                     "no_range_sentence_in_cited_sources", "not_exact_cited_sentence"})


class AgentOutputError(ValueError):
    def __init__(self, code: str, fields=(), *, rules=(), evidence_reason: str | None = None):
        if code not in ERRORS: raise ValueError("unknown_output_error_code")
        if evidence_reason is not None and (code != "report_range_claim_without_exact_source"
                                            or evidence_reason not in REPORT_EVIDENCE_REASONS):
            raise ValueError("unknown_evidence_reason")
        self.code = code
        self.evidence_reason = evidence_reason
        self.stage, self.repairable = ERRORS[code]
        self.fields = tuple(dict.fromkeys(field if field in SAFE_FIELDS else "unknown_field" for field in fields))[:6]
        self.rules = tuple(dict.fromkeys(rule if rule in SAFE_RULES else "unknown_rule" for rule in rules))[:4]
        super().__init__(code)

    def diagnostic(self):
        result = {"code":self.code,"stage":self.stage,"fields":list(self.fields),"repairable":self.repairable}
        if self.rules: result["rules"] = list(self.rules)
        if self.evidence_reason is not None: result["evidence_reason"] = self.evidence_reason
        return result


def schema_error(exc: ValidationError) -> AgentOutputError:
    # Pydantic loc may include an arbitrary extra key; discard it along with inputs/context/messages.
    fields,rules = [],[]
    for item in exc.errors(include_input=False,include_context=False,include_url=False)[:4]:
        fields.extend(part for part in item["loc"] if isinstance(part,str))
        if item["type"] in ROOT_RULE_FIELDS:
            fields.extend(ROOT_RULE_FIELDS[item["type"]])
            rules.append(item["type"])
        else:
            rules.append(FIELD_RULES.get(item["type"],"schema_validation_failed"))
    return AgentOutputError("invalid_output_shape",fields,rules=rules)
