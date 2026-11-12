"""Context-aware fake model and multi-turn request helper."""
import json
from medagent.core.contracts import Question
from support.harness import Model


class ContextModel(Model):
    def __init__(self, updates=None, tasks=None):
        self.updates, self.tasks, self.inputs = updates, tasks or [], []

    async def json(self, stage, system, content, budget=900):
        if stage != "understanding":
            return await super().json(stage, system, content, budget)
        payload = json.loads(content)
        self.inputs.append(payload)
        # Like the real model, the fake one reads the history to follow a short correction or reference.
        scores = await super().json(stage, system, content, budget)
        candidates = self.updates(payload) if self.updates else []
        return {**scores, "tasks":self.tasks, "updates":[{"turn_id":payload["current"]["turn_id"], **item} for item in candidates]}


async def ask(runtime, text, user="synthetic", session="dialogue", **extra):
    return await runtime.chat(Question(text=text, user=user, session=session, **extra))
