from dataclasses import replace

from medagent.appointments.catalog import HospitalCatalog
from medagent.appointments.store import AppointmentStore
from medagent.comparison.runner import ArchitectureIsolation
from medagent.core.config import ROOT
from medagent.evaluation.contracts import EvalLimits
from medagent.evaluation.budget import ModelBudget
from medagent.evaluation.isolation import EvaluationIsolation
from support.harness import fixture, run

def book(store):
    from datetime import timedelta
    for offset in range(1, 15):
        found = store.catalog.slots("demo-general", "neurology", store.today() + timedelta(days=offset), "am", store.today())
        if found:
            draft, token = store.propose_create(user="eval-user", session="s", subject="self", slot_id=found[0].slot_id)
            return store.confirm(user="eval-user", draft_id=draft.draft_id, token=token)


def file_source(tmp_path):
    async def build():
        path = str(tmp_path/"appointments.sqlite3")
        source = await fixture(preview=False, appointments=AppointmentStore(HospitalCatalog.load(ROOT/"resources"/"mock-hospital.json"), path))
        source.settings = replace(source.settings, appointment_db=path)
        return source, path
    return run(build())


def test_evaluation_isolation_never_writes_the_real_store(tmp_path):
    source, path = file_source(tmp_path)

    async def scenario():
        isolation = EvaluationIsolation(source, "appointments-isolated", ModelBudget(EvalLimits()))
        runtime = await isolation.create()
        try:
            book(runtime.appointments)
        finally:
            await isolation.cleanup()
    run(scenario())
    assert AppointmentStore(source.appointments.catalog, path).bookings("eval-user") == []


def test_comparison_arms_get_their_own_store(tmp_path):
    source, path = file_source(tmp_path)

    async def scenario():
        isolation = ArchitectureIsolation(source, "appointments-arms", None)
        await isolation.create()
        try:
            first = await isolation.arm_branch("multi_agent", None)
            book(first.appointments)
            second = await isolation.arm_branch("single_agent", None)
            assert second.appointments.bookings("eval-user") == []
        finally:
            await isolation.cleanup()
    run(scenario())
    assert AppointmentStore(source.appointments.catalog, path).bookings("eval-user") == []
