"""Measure target-turn provider usage; preparation calls are reported separately."""
import argparse
import asyncio
import json
import uuid
from pathlib import Path
import httpx

CASES = [
    ["我想整理去看医生之前的信息。", "我已经有一份体检报告，也列好了现在使用的药品。", "请总结我还应该记录和准备哪些就诊信息。"],
    ["我在学习如何阅读体检报告。", "我知道报告里有项目、结果、单位和参考范围。", "单项结果在参考范围之外一定表示有病吗？"]
]


async def compare(url, output):
    rows = []
    async with httpx.AsyncClient(timeout=300) as client:
        for case_id, messages in enumerate(CASES):
            for mode in ["selective", "full"]:
                session, preparation = "", []
                user = f"benchmark-{uuid.uuid4()}"
                for index, text in enumerate(messages):
                    response = await client.post(url.rstrip("/")+"/chat",json={"text":text,"user":user,"session":session,"context":mode})
                    response.raise_for_status()
                    reply=response.json()
                    session=reply["session"]
                    if index < len(messages)-1:
                        preparation.append(reply["usage"])
                rows.append({"case":case_id,"mode":mode,"target_usage":reply["usage"],"preparation_usage":preparation,
                             "reply":reply,"runtime_check":not reply["degraded"]})
    totals = {mode:[row["target_usage"]["input_tokens"] for row in rows if row["mode"]==mode] for mode in ["selective","full"]}
    means = {mode:sum(values)/len(values) if all(isinstance(value,int) for value in values) else None for mode,values in totals.items()}
    reduction = 1-means["selective"]/means["full"] if means["selective"] is not None and means["full"] else None
    report={"dataset":"context-dialogue-v1","mean_input_tokens":means,"input_reduction":reduction,"quality_equivalence":"not_established","rows":rows}
    target=Path(output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(target)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--url",default="http://127.0.0.1:8000")
    parser.add_argument("--output",default="artifacts/context.json")
    args=parser.parse_args()
    asyncio.run(compare(args.url,args.output))
