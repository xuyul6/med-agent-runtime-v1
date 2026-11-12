"""Offline drug-name recognition probe: extraction and primary role over templated requests."""
import asyncio
import json
from medagent.core.config import ROOT
from support.harness import fixture
from support.multiturn import ContextModel, ask

TEMPLATES = ["想查{}的公开资料", "{}有什么禁忌", "我在吃{}，想了解说明书", "请介绍一下{}这个药"]


def names():
    body = json.loads((ROOT/"resources/medication-names.json").read_text())
    return [(item["name"], item["name"]) for item in body["medications"]] + \
           [(alias, item["name"]) for item in body["medications"] for alias in item["aliases"] if alias.isascii()]


async def probe():
    runtime = await fixture(ContextModel())
    rows = []
    for index, (written, canonical) in enumerate(names()):
        for template in TEMPLATES if not written.isascii() else TEMPLATES[:1]:
            text = template.format(written)
            reply = await ask(runtime, text, user=f"probe-{index}-{template}", session="p")
            fact = reply.intent.entities.get("medication_name")
            rows.append({"text":text, "canonical":canonical, "extracted":fact.value if fact else None,
                         "role":reply.primary_agent})
    return rows


if __name__ == "__main__":
    rows = asyncio.run(probe())
    extracted = sum(row["extracted"] is not None for row in rows)
    routed = sum(row["role"] == "drug_reference" for row in rows)
    print(f"{len(rows)} requests; medication_name extracted {extracted}; routed to drug_reference {routed}")
