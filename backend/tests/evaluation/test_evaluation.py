import dataclasses
import asyncio
import copy
from dataclasses import replace
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from medagent.app import create_app
from medagent.core.contracts import Risk
from medagent.evaluation.runner import evaluate, resolve_cases
from medagent.evaluation.budget import BudgetedModel, EvaluationBudgetExceeded, ModelBudget
from medagent.evaluation.contracts import EvaluationRequest, EvalLimits, JudgeScore
from medagent.evaluation.isolation import EvaluationIsolation
from medagent.evaluation.metrics import BaselineError, BaselineStore, classification_metrics, compare_baseline
from medagent.memory import SessionMemory
from medagent.core.observability import CURRENT, RunLog, event
from support.harness import captured_deadlines, expire, fixture, run
from support.evaluation import JudgeModel


def request(turns=None, **kwargs):
    return EvaluationRequest(intent_cases=[],dialog_cases=[{'id':'dialog','turns':turns or ['你好']}],**kwargs)


def test_fixed_label_metrics_account_for_errors_zero_support_and_math():
    metrics = classification_metrics(['a','a','b','b'],['a','b','b',None],['a','b'])
    assert metrics['accuracy'] == .5 and metrics['failures'] == 1
    assert metrics['per_class']['a']['precision'] == 1 and metrics['per_class']['a']['recall'] == .5
    assert metrics['per_class']['a']['f1'] == pytest.approx(2/3)
    assert metrics['per_class']['b']['f1'] == .5
    assert metrics['macro_f1'] == pytest.approx(7/12)
    assert metrics['confusion_matrix']['b']['__error__'] == 1
    one = classification_metrics(['a'],['a'],['a','b'])
    assert one['macro_f1'] == .5
    empty = classification_metrics([],[],['a','b'])
    assert empty['accuracy'] is None and empty['macro_f1'] is None


def test_default_dataset_retains_six_cases_and_original_medication_failure():
    version,intents,dialogs = resolve_cases(EvaluationRequest())
    assert version == 'medical-runtime-eval-v3' and len(intents)+len(dialogs) == 39
    original = json.loads((Path(__file__).parents[2]/'resources/evaluation.json').read_text())['cases']
    assert len(original) == 6
    assert [case.turns[0].message for case in dialogs[:6]] == [row['text'] for row in original]
    assert len({case.id for case in [*intents,*dialogs]}) == 39
    assert any(len(case.turns) >= 6 for case in dialogs)
    assert {'memory_unavailable','tool_timeout','summary_failure'} <= {fault.kind for case in dialogs for fault in case.faults}
    assert resolve_cases(EvaluationRequest(intent_cases=[],dialog_cases=[]))[1:] == ([],[])


@pytest.mark.parametrize('body',[
    {'dialog_case':[]}, {'intent_cases':None}, {'dialog_cases':None},
    {'intent_cases':[{'message':'x','expected':'refund'}]},
    {'intent_cases':[{'message':'x','expected':'general','expected_risk':'CRITICAL'}]},
    {'intent_cases':[{'message':'x','expected':'urgent','expected_risk':'ROUTINE'}]},
    {'dialog_cases':[{'turns':[]}]}, {'dialog_cases':[{'turns':['x']*9}]},
    {'dialog_cases':[{'turns':['x'],'unexpected':True}]},
    {'dialog_cases':[{'turns':['x'],'faults':[{'kind':'delete_files'}]}]},
    {'dialog_cases':[{'turns':['x'],'faults':[{'kind':'tool_timeout','turn':2}]}]},
    {'baseline':{'compare_to':'../private'}}, {'limits':{'max_model_calls':401}},
    {'limits':{'concurrency':2}}, {'limits':{'timeout_seconds':0}},
    {'dialog_cases':[{'turns':[{'message':'x','expected':{'missing_fields':['password']}}]}]},
    {'intent_cases':[{'id':'same','message':'x','expected':'general'}]*2},
])
def test_invalid_evaluation_request_is_rejected(body):
    with pytest.raises(ValidationError): EvaluationRequest.model_validate(body)


def test_custom_string_turns_are_normalized_with_stable_case_ids():
    parsed = EvaluationRequest.model_validate({'intent_cases':[],'dialog_cases':[{'turns':['你好','谢谢']}]})
    assert parsed.dialog_cases[0].id == 'dialog-1'
    assert parsed.dialog_cases[0].turns[1].message == '谢谢'


def test_long_valid_case_id_does_not_overflow_generated_user_scope():
    async def scenario():
        runtime = await fixture(JudgeModel())
        body = EvaluationRequest(intent_cases=[],judge=False,dialog_cases=[{'id':'c'*64,'turns':['现在呼吸困难']}])
        result = await evaluate(runtime,body)
        assert result['complete'] and result['passed'] == 1
    run(scenario())


def test_explicit_empty_arrays_execute_nothing_and_report_unknown_quality():
    async def scenario():
        runtime = await fixture(JudgeModel())
        result = await evaluate(runtime,EvaluationRequest(intent_cases=[],dialog_cases=[],judge=False))
        assert result['complete'] and result['total'] == result['executed'] == 0
        assert result['pass_rate'] is None and result['metrics']['quality_mean'] is None
        assert result['budget']['attempted_calls'] == 0 and not runtime.monitor.requests
    run(scenario())


def test_custom_dialogue_shares_real_memory_and_original_field_provenance():
    async def scenario():
        runtime = await fixture(JudgeModel())
        result = await evaluate(runtime,request([
            {'message':'查询布洛芬公开资料。','expected':{'entities':{'medication_name':'布洛芬'}}},
            {'message':'这个药的资料呢？','expected':{'entities':{'medication_name':'布洛芬'}}}]))
        assert result['complete'] and result['passed'] == 1
        turns = result['cases'][0]['turns']
        assert turns[0]['reply']['session'] == turns[1]['reply']['session']
        fact = turns[1]['reply']['intent']['entities']['medication_name']
        assert fact['source']['turn_id'] == turns[0]['reply']['request_id'] and fact['source']['origin'] == 'history_user'
        assert result['mode'] == 'offline' and result['judge']['scored'] == 2
        assert result['usage']['runtime']['input_tokens'] is None  # Fake model did not emit provider usage.
        assert result['usage']['judge']['attempted_calls'] == 2
    run(scenario())


def test_eval_isolates_monitor_cache_risk_subjects_tools_and_user_memory():
    async def scenario():
        runtime = await fixture(JudgeModel())
        runtime.monitor.record('drug_reference.1',False,100)
        runtime.tools.failures['lookup_reference'] = 1
        runtime.index.cache['production'] = (float('inf'),[])
        runtime.memory.local['production'] = {'messages':[],'summary':'private marker'}
        before = (copy.deepcopy(runtime.monitor.snapshot()),copy.deepcopy(runtime.index.cache),copy.deepcopy(runtime.memory.local),dict(runtime.tools.failures))
        body = request(['查询布洛芬公开资料。'],judge=False)
        result = await evaluate(runtime,body)
        assert result['complete']
        assert runtime.monitor.snapshot() == before[0] and runtime.index.cache == before[1]
        assert runtime.memory.local == before[2] and runtime.tools.failures == before[3]
        assert not runtime.subjects.active and not runtime.risk.hints
        assert not result['cleanup_errors']
    run(scenario())


def test_eval_uses_namespaced_redis_and_deletes_only_its_own_keys():
    async def scenario():
        class Redis:
            def __init__(self): self.rows,self.reads,self.writes,self.deleted = {'production':'private'},[],[],[]
            async def get(self,key): self.reads.append(key); return self.rows.get(key)
            async def set(self,key,value,ex): self.rows[key]=value; self.writes.append((key,ex))
            async def delete(self,*keys):
                self.deleted.extend(keys)
                for key in keys: self.rows.pop(key,None)
        redis = Redis()
        runtime = await fixture(JudgeModel())
        runtime.memory = SessionMemory(redis=redis,model=runtime.model)
        result = await evaluate(runtime,request(['查询布洛芬公开资料。','这个药呢？'],judge=False))
        assert result['complete'] and redis.rows == {'production':'private'}
        assert redis.writes and all(key.startswith('mar2:eval:'+result['run_id']) and ttl<=3600 for key,ttl in redis.writes)
        assert set(redis.deleted) == {key for key,_ in redis.writes}
        assert all(key.startswith('mar2:eval:') for key in redis.reads)
    run(scenario())


def test_summary_and_episode_recall_run_through_same_session_memory_interface():
    async def scenario():
        runtime = await fixture(JudgeModel())
        result = await evaluate(runtime,request(['查询布洛芬公开资料。']+['这个药的资料呢？']*6,judge=False))
        assert result['complete']
        turns = result['cases'][0]['turns']
        assert any(event['event']=='memory_compressed' for turn in turns for event in turn['trace']['events'])
        assert turns[-1]['reply']['intent']['entities']['medication_name']['value'] == '布洛芬'
        assert not runtime.memory.local
    run(scenario())


def test_subject_switch_and_case_scope_are_isolated_during_evaluation():
    async def scenario():
        runtime = await fixture(JudgeModel())
        body = EvaluationRequest(intent_cases=[],judge=False,dialog_cases=[{'id':'first','turns':[
            '查询布洛芬公开资料。',{'message':'现在给我妈妈问，这个药呢？','expected':{'subject':'mother','missing_fields':['medication_name']}}]},
            {'id':'second','turns':[{'message':'这个药呢？','expected':{'missing_fields':['medication_name']}}]}])
        result = await evaluate(runtime,body)
        assert result['passed'] == 2
        assert 'medication_name' not in result['cases'][1]['turns'][0]['reply']['intent']['entities']
    run(scenario())


@pytest.mark.parametrize('fault',['memory_unavailable','tool_timeout','tool_invalid_result','agent_invalid_output','index_unavailable'])
def test_faults_are_exercised_only_in_isolation_and_reset_between_cases(fault):
    async def scenario():
        runtime = await fixture(JudgeModel(),preview=fault != 'index_unavailable')
        body = EvaluationRequest(intent_cases=[],judge=False,dialog_cases=[
            {'id':'fault','turns':[{'message':'查询布洛芬公开资料。','expected':{'degraded':True}}],'faults':[{'kind':fault}]},
            {'id':'normal','turns':[{'message':'查询布洛芬公开资料。','expected':{'degraded':False}}]}])
        result = await evaluate(runtime,body)
        assert result['complete'] and result['passed'] == 2
        assert not runtime.tools.failures and not runtime.monitor.requests
        assert result['cases'][0]['turns'][0]['reply']['degraded']
        assert not result['cases'][1]['turns'][0]['reply']['degraded']
    run(scenario())


def test_intent_classifier_and_urgent_gate_have_different_denominators():
    async def scenario():
        runtime = await fixture(JudgeModel())
        body = EvaluationRequest(dialog_cases=[],judge=False,intent_cases=[
            {'message':'你好','expected':'general'}, {'message':'现在呼吸困难','expected':'urgent','expected_risk':'CRITICAL'}])
        result = await evaluate(runtime,body)
        assert result['passed'] == 2
        assert result['classification']['classifier']['total'] == 1
        assert result['classification']['classifier']['macro_f1'] == .2
        assert result['classification']['risk_gated']['total'] == 2
        assert result['cases'][1]['runtime_usage']['attempted_calls'] == 0
    run(scenario())


def test_language_fault_can_exercise_memory_and_clarification_without_forwarded_model_calls():
    async def scenario():
        runtime = await fixture(JudgeModel())
        body = EvaluationRequest(intent_cases=[],judge=False,dialog_cases=[{'turns':[
            {'message':'我想查药，但药名不清楚。','expected':{'role':'drug_reference','status':'needs_clarification'}},
            {'message':'还是不清楚药名。','expected':{'role':'drug_reference','status':'needs_clarification'}}],
            'faults':[{'kind':'language_unavailable'}]}])
        result = await evaluate(runtime,body)
        assert result['complete'] and result['passed'] == 1 and result['budget']['attempted_calls'] == 0
        assert result['isolation']['temporary_database_removed']
    run(scenario())


def test_classifier_exception_is_counted_as_error_in_accuracy_denominator():
    async def scenario():
        class Broken(EvaluationIsolation):
            def new_case(self):
                runtime = super().new_case()
                async def fail(*args,**kwargs): raise RuntimeError('synthetic classifier failure')
                runtime.router.understand = fail
                return runtime
        runtime = await fixture(JudgeModel())
        body = EvaluationRequest(dialog_cases=[],judge=False,intent_cases=[{'message':'你好','expected':'general'}])
        result = await evaluate(runtime,body,isolation_factory=Broken)
        assert result['classification']['classifier']['failures'] == 1
        assert result['classification']['classifier']['accuracy'] == 0
        assert result['recommendations'][0]['case_id'] == 'intent-1'
    run(scenario())


def test_false_positive_risk_gate_is_a_gated_error_and_not_fabricated_classifier_prediction():
    async def scenario():
        class OveractiveGate(EvaluationIsolation):
            def new_case(self):
                runtime = super().new_case()
                runtime.risk.screen = lambda *args:Risk(level='CRITICAL')
                return runtime
        body = EvaluationRequest(dialog_cases=[],judge=False,intent_cases=[{'message':'你好','expected':'general'}])
        result = await evaluate(await fixture(JudgeModel()),body,isolation_factory=OveractiveGate)
        assert result['classification']['classifier']['accuracy'] is None
        assert result['classification']['classifier']['skipped_by_risk_gate'] == 1
        assert result['metrics']['classifier_coverage'] == 0
        assert result['classification']['risk_gated']['accuracy'] == 0 and result['passed'] == 0
    run(scenario())


@pytest.mark.parametrize('score',[True,-.1,1.1,float('nan'),float('inf'),'0.8'])
def test_invalid_judge_values_are_rejected(score):
    with pytest.raises(ValidationError):
        JudgeScore.model_validate({**{key:score for key in ['relevance','accuracy','completeness','helpfulness']},'reason':'synthetic'})


def test_judge_failures_stay_missing_and_do_not_disappear_from_pass_rate():
    async def scenario():
        runtime = await fixture(JudgeModel(broken=True))
        result = await evaluate(runtime,request(['你好']))
        assert result['rule_passed'] == 1 and result['passed'] == 0 and result['complete']
        assert result['judge']['errors'] == result['judge']['missing'] == 1
        assert result['metrics']['quality_mean'] is None and result['metrics']['judge_coverage'] == 0
        assert result['cases'][0]['judge_error'] == 'ConnectionError'
        assert result['recommendations'][0]['trace_id']
    run(scenario())


def test_high_judge_score_cannot_override_new_safety_violation():
    async def scenario():
        class Unsafe(EvaluationIsolation):
            def new_case(self):
                runtime = super().new_case()
                chat = runtime.chat
                async def unsafe(question):
                    reply = await chat(question)
                    reply.answer.text = '已为你预约专家门诊。'
                    return reply
                runtime.chat = unsafe
                return runtime
        runtime = await fixture(JudgeModel())
        result = await evaluate(runtime,request(['你好']),isolation_factory=Unsafe)
        assert result['metrics']['quality_mean'] == 1 and result['passed'] == 0
        assert any(row['gate']=='output_boundaries' for row in result['safety_failures'])
    run(scenario())


def test_low_judge_recommendations_have_case_and_score_evidence():
    async def scenario():
        result = await evaluate(await fixture(JudgeModel(score=.5)),request(['你好']))
        assert result['passed'] == 0
        assert any('judge_below_threshold:' in item for item in result['recommendations'][0]['evidence'])
    run(scenario())


def test_budget_limits_stop_remaining_cases_without_erasing_attempted_failures():
    async def scenario():
        runtime = await fixture(JudgeModel())
        body = EvaluationRequest(judge=False,dialog_cases=[],intent_cases=[{'message':'你好','expected':'general'} for _ in range(3)],limits={'max_model_calls':1})
        result = await evaluate(runtime,body)
        assert not result['complete'] and result['stop_reason'] == 'max_model_calls'
        assert result['total'] == 3 and result['executed'] == 2 and result['not_run'] == ['intent-3']
        assert result['budget']['attempted_calls'] == 1 and result['cases'][-1]['passed'] is False
    run(scenario())


def test_requested_output_budget_is_reserved_before_forwarding_and_never_exceeded():
    async def scenario():
        class Base:
            async def json(self,stage,system,content,budget):
                assert budget == 5
                return {}
        budget = ModelBudget(EvalLimits(max_output_tokens=5))
        model = BudgetedModel(Base(),budget)
        await model.json('test','x','x',budget=900)
        with pytest.raises(EvaluationBudgetExceeded): await model.json('test','x','x')
        assert budget.reserved_output == 5 and budget.calls == 1
    run(scenario())


@pytest.mark.parametrize('tokens',[True,-1,None])
def test_invalid_provider_usage_stays_unknown(tokens):
    log = RunLog()
    log.add('model',input_tokens=tokens,output_tokens=1)
    assert log.usage()['input_tokens'] is None and log.usage()['measurement'] == 'unavailable'


def test_deadline_preserves_partial_turn_results_and_cleans_up_isolated_state(monkeypatch):
    deadlines = captured_deadlines(monkeypatch)
    async def scenario():
        class Slow(JudgeModel):
            async def json(self,stage,system,content,budget=900):
                if stage == 'evaluation_judge':
                    # The run's deadline passes once the first turn has its reply, however long the setup took.
                    expire(deadlines[0])
                    await asyncio.sleep(10)
                return await super().json(stage,system,content,budget)
        runtime = await fixture(Slow())
        result = await evaluate(runtime,request(['你好','谢谢'],limits={'timeout_seconds':60}))
        assert not result['complete'] and result['stop_reason'] == 'timeout'
        assert result['cases'][0]['turns'][0]['reply']
        assert result['cases'][0]['not_run_turns'] == [2]
        assert not result['cleanup_errors'] and not runtime.monitor.requests
    run(scenario())


def test_provider_usage_is_separated_between_runtime_and_judge_with_concurrent_calls():
    async def scenario():
        class Measured(JudgeModel):
            async def json(self,stage,*args,**kwargs):
                event('model',stage=stage,input_tokens=7,output_tokens=3)
                return await super().json(stage,*args,**kwargs)
            async def message(self,stage,*args,**kwargs):
                event('model',stage=stage,input_tokens=11,output_tokens=5)
                return await super().message(stage,*args,**kwargs)
        runtime = await fixture(Measured())
        result = await evaluate(runtime,request(['你好']))
        assert result['usage']['runtime']['input_tokens'] == 18
        assert result['usage']['judge']['input_tokens'] == 7
        assert result['usage']['runtime']['attempted_calls'] == 2
        assert result['usage']['judge']['attempted_calls'] == 1
    run(scenario())


def test_concurrent_model_children_do_not_double_count_parent_log_usage():
    async def scenario():
        class Measured:
            async def json(self,stage,*args,**kwargs):
                await asyncio.sleep(0)
                event('model',stage=stage,input_tokens=10,output_tokens=2)
                return {}
        budget = ModelBudget(EvalLimits())
        model = BudgetedModel(Measured(),budget)
        parent = RunLog(); token = CURRENT.set(parent)
        try: await asyncio.gather(model.json('first','x','x'),model.json('second','x','x'))
        finally: CURRENT.reset(token)
        assert parent.usage()['calls'] == 2 and parent.usage()['input_tokens'] == 20
        assert len(budget.entries) == 2
    run(scenario())


def test_known_judge_usage_is_preserved_when_a_later_judge_call_has_unknown_usage():
    async def scenario():
        class Mixed(JudgeModel):
            def __init__(self): super().__init__(); self.judges = 0
            async def json(self,stage,*args,**kwargs):
                if stage == 'evaluation_judge':
                    self.judges += 1
                    if self.judges == 2: raise ConnectionError('synthetic')
                    event('model',stage=stage,input_tokens=7,output_tokens=3)
                return await super().json(stage,*args,**kwargs)
        result = await evaluate(await fixture(Mixed()),request(['你好','谢谢']))
        turns = result['cases'][0]['turns']
        assert result['usage']['judge']['input_tokens'] is None
        assert turns[0]['judge_usage']['input_tokens'] == 7 and turns[1]['judge_usage']['input_tokens'] is None
        assert turns[0]['judge_traces'][0]['events'][0]['input_tokens'] == 7
        assert not result['passed']
    run(scenario())


def test_cancellation_cleans_only_owned_database_and_leaves_no_sampler():
    async def scenario():
        started = asyncio.Event()
        class Slow(JudgeModel):
            async def json(self,stage,*args,**kwargs):
                if stage == 'evaluation_judge':
                    started.set(); await asyncio.sleep(10)
                return await super().json(stage,*args,**kwargs)
        owners = []
        class Captured(EvaluationIsolation):
            def __init__(self,*args): super().__init__(*args); owners.append(self)
        runtime = await fixture(Slow())
        task = asyncio.create_task(evaluate(runtime,request(['你好']),isolation_factory=Captured))
        await started.wait()
        owned = owners[0]
        directory = Path(owned.directory.name)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert not directory.exists() and owned.runtime.monitor._task is None
        assert not runtime.memory.local and not runtime.monitor.requests
    run(scenario())


def test_summary_and_judge_faults_are_recorded_without_synthetic_success_scores():
    async def scenario():
        runtime = await fixture(JudgeModel())
        body = EvaluationRequest(intent_cases=[],dialog_cases=[
            {'id':'summary','turns':['查询布洛芬公开资料。']+['这个药呢？']*4+[{'message':'这个药呢？','expected':{'degraded':True}}],
             'faults':[{'kind':'summary_failure','turn':6}]},
            {'id':'judge','turns':['你好'],'faults':[{'kind':'judge_failure'}]}])
        result = await evaluate(runtime,body)
        assert result['complete'] and result['cases'][0]['passed']
        assert result['cases'][1]['quality'] is None and result['cases'][1]['judge_error']
        assert result['passed'] == 1
    run(scenario())


def baseline_report(value=.8):
    return {'schema_version':'evaluation-2','complete':True,'metadata':{'dataset_hash':'d','metric_version':'m','policy_hash':'p','model':'old'},
            'metrics':{'quality_mean':value},'safety_failures':[]}


@pytest.mark.parametrize('current,regresses',[(.76,False),(.7599,True),(.8,False),(.9,False)])
def test_baseline_relative_drop_must_be_strictly_greater_than_five_percent(current,regresses):
    comparison = compare_baseline(baseline_report(current),baseline_report(.8))
    assert bool(comparison['regressions']) is regresses


def test_baseline_zero_missing_safety_and_experiment_variables_are_separate():
    baseline = baseline_report(0)
    current = baseline_report(None)
    assert compare_baseline(current,baseline)['metrics'][0]['status'] == 'missing'
    current['metrics']['quality_mean'] = 0
    assert compare_baseline(current,baseline)['metrics'][0]['status'] == 'zero_baseline'
    current['safety_failures'] = [{'case_id':'case','turn':1,'gate':'urgent_short_circuit'}]
    current['metadata']['model'] = 'new'
    comparison = compare_baseline(current,baseline)
    assert comparison['regressions'][0]['kind'] == 'new_safety_failure'
    assert comparison['variables'][0]['field'] == 'model'
    current['metadata']['dataset_hash'] = 'different'
    with pytest.raises(BaselineError): compare_baseline(current,baseline)


def test_saved_baselines_are_immutable_and_dataset_change_rejected_before_calls(tmp_path):
    async def scenario():
        store = BaselineStore(tmp_path)
        runtime = await fixture(JudgeModel())
        result = await evaluate(runtime,request(['你好'],baseline={'save_as':'first'}),baseline_store=store)
        assert result['baseline']['saved']
        rechecked = await evaluate(runtime,request(['你好'],baseline={'compare_to':'first'}),baseline_store=store)
        assert rechecked['comparison']['status'] == 'compared' and not rechecked['regressions']
        with pytest.raises(BaselineError):
            await evaluate(runtime,request(['谢谢'],baseline={'compare_to':'first'}),baseline_store=store)
        with pytest.raises(BaselineError): store.save('first',result)
        assert not list(tmp_path.glob('.baseline-*'))
        assert not runtime.monitor.requests
    run(scenario())


def test_passing_case_can_still_have_evidence_backed_quality_regression(tmp_path):
    async def scenario():
        store = BaselineStore(tmp_path)
        await evaluate(await fixture(JudgeModel()),request(['你好'],baseline={'save_as':'first'}),baseline_store=store)
        result = await evaluate(await fixture(JudgeModel(score=.8)),request(['你好'],baseline={'compare_to':'first'}),baseline_store=store)
        assert result['passed'] == 1 and result['regressions']
        assert any(row['case_id']=='dialog' and row['trace_id'] for row in result['recommendations'])
    run(scenario())


def test_incomplete_and_symlink_baselines_cannot_be_used(tmp_path):
    store = BaselineStore(tmp_path)
    report = baseline_report(); report['complete'] = False
    assert store.save('incomplete',report)['saved'] is False
    (tmp_path/'unsafe.json').symlink_to(tmp_path/'target.json')
    with pytest.raises(BaselineError): store.load('unsafe')


@pytest.mark.parametrize('changes',[{'safety_failures':[{}]},{'cases':[{}]},
    {'cases':[{'id':'x','turns':[{'turn':1,'quality':{'mean':True}}]}]}])
def test_malformed_baseline_case_evidence_is_rejected_before_execution(tmp_path,changes):
    report = {**baseline_report(),**changes}
    (tmp_path/'bad.json').write_text(json.dumps(report))
    with pytest.raises(BaselineError): BaselineStore(tmp_path).load('bad')


def test_offline_baseline_is_not_compared_to_online_model_configuration(tmp_path):
    async def scenario():
        runtime = await fixture(JudgeModel())
        store = BaselineStore(tmp_path)
        await evaluate(runtime,request(['你好'],baseline={'save_as':'offline'}),baseline_store=store)
        runtime.settings = replace(runtime.settings,preview=False)
        with pytest.raises(BaselineError):
            await evaluate(runtime,request(['你好'],baseline={'compare_to':'offline'}),baseline_store=store)
    run(scenario())


def test_new_safety_failure_is_reported_even_when_other_cases_were_not_run():
    baseline,current = baseline_report(),baseline_report()
    current['complete'] = False
    current['safety_failures'] = [{'case_id':'case','turn':1,'gate':'urgent_short_circuit'}]
    result = compare_baseline(current,baseline)
    assert result['status'] == 'incomplete_run' and result['regressions'][0]['kind'] == 'new_safety_failure'
    assert result['metrics'] == []


def test_http_custom_cases_empty_arrays_validation_and_baseline_conflicts(tmp_path):
    runtime = run(fixture(JudgeModel()))
    app = create_app(settings=dataclasses.replace(runtime.settings, developer_tools=True),runtime=runtime)
    app.state.baseline_store = BaselineStore(tmp_path)
    with TestClient(app) as client:
        response = client.post('/eval/run',json={'judge':False,'intent_cases':[{'message':'你好','expected':'general'}],'dialog_cases':[]})
        assert response.status_code == 200 and response.json()['executed'] == 1
        assert response.json()['classification']['classifier']['accuracy'] == 1
        assert client.post('/eval/run',json={'judge':False,'intent_cases':[],'dialog_cases':[]}).json()['total'] == 0
        assert client.post('/eval/run',json={'judge':False,'unknown':1}).status_code == 422
        assert client.post('/eval/run',json={'judge':False,'baseline':{'compare_to':'../x'}}).status_code == 422
        assert client.post('/eval/run',json={'judge':False,'intent_cases':[],'dialog_cases':[],'baseline':{'compare_to':'missing'}}).status_code == 409
        assert not runtime.monitor.requests


def test_http_rejects_another_evaluation_while_one_is_active():
    runtime = run(fixture(JudgeModel()))
    app = create_app(settings=dataclasses.replace(runtime.settings, developer_tools=True),runtime=runtime)
    with TestClient(app) as client:
        client.portal.call(app.state.evaluation_lock.acquire)
        try: assert client.post('/eval/run',json={'judge':False,'intent_cases':[],'dialog_cases':[]}).status_code == 409
        finally: client.portal.call(app.state.evaluation_lock.release)


def test_cli_forwards_request_and_preserves_existing_reports(tmp_path,monkeypatch):
    import httpx
    from medagent.evaluate import run as cli_run
    real_client = httpx.AsyncClient
    body = {'judge':False,'intent_cases':[],'dialog_cases':[]}
    seen = []
    async def respond(req):
        if req.url.path == '/eval/run':
            seen.append(json.loads(req.content))
            return httpx.Response(200,json={'passed':0,'total':0,'executed':0,'complete':True})
        return httpx.Response(200,json={'preview':True,'developer_tools':True})
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:real_client(transport=httpx.MockTransport(respond)))
    target = tmp_path/'report.json'
    run(cli_run('http://synthetic',str(target),body))
    assert seen == [body] and target.exists()
    with pytest.raises(FileExistsError): run(cli_run('http://synthetic',str(target),body))
    assert seen == [body]
