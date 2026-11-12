"""Deterministic Judge stand-in for evaluation tests."""
from support.multiturn import ContextModel


class JudgeModel(ContextModel):
    def __init__(self, score=1, broken=False):
        super().__init__()
        self.score,self.broken = score,broken

    async def json(self, stage, system, content, budget=900):
        if stage == 'evaluation_judge':
            if self.broken: raise ConnectionError('synthetic')
            return {**{key:self.score for key in ['relevance','accuracy','completeness','helpfulness']},'reason':'合成 Judge，仅用于框架测试。'}
        return await super().json(stage,system,content,budget)
