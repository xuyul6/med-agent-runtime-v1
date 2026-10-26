import argparse
import asyncio
import json
from pathlib import Path
import httpx


async def run(url, output):
    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(url.rstrip("/")+"/eval/run")
        response.raise_for_status()
        result = response.json()
        health = (await client.get(url.rstrip("/")+"/health")).json()
    result["environment"] = health
    target = Path(output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(f'{result["passed"]}/{result["total"]} quality+rules; {result["rule_passed"]}/{result["total"]} rules. Saved: {target}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",default="http://127.0.0.1:8000")
    parser.add_argument("--output",default="artifacts/evaluation.json")
    arguments = parser.parse_args()
    asyncio.run(run(arguments.url,arguments.output))
