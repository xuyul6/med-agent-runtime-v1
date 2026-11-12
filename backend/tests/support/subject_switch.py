"""Stage-counting fake model and clarification text helper."""
import json
from support.multiturn import ContextModel


class CountStages(ContextModel):
    def __init__(self): super().__init__(); self.stages = []
    async def json(self, stage, *args, **kwargs):
        self.stages.append(stage)
        return await super().json(stage,*args,**kwargs)
    async def message(self, stage, *args, **kwargs):
        self.stages.append(stage)
        return await super().message(stage,*args,**kwargs)


def guidance_text(reply):
    return json.dumps(reply.answer.model_dump(),ensure_ascii=False)
