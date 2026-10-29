"""Build an evaluation snapshot from a previously exported public catalog."""
import argparse
import json
from pathlib import Path

from .contracts import Reference
from .evaluation import reference_snapshot_hash
from .evaluation_contracts import ReferenceSnapshot


def make_reference_snapshot(records, keys):
    catalog = {record.key:record for record in records}
    if len(catalog) != len(records): raise ValueError("duplicate_catalog_key")
    if len(set(keys)) != len(keys): raise ValueError("duplicate_reference_key")
    if set(keys)-set(catalog): raise ValueError("snapshot_reference_missing")
    return ReferenceSnapshot(keys=keys, sha256=reference_snapshot_hash([catalog[key] for key in keys])).model_dump(mode="json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, help="JSON response exported from GET /knowledge")
    parser.add_argument("--key", action="append", default=[], help="include a reference key; repeat as needed")
    args = parser.parse_args()
    data = json.loads(Path(args.catalog).read_text())
    records = [Reference.model_validate(row) for row in (data if isinstance(data,list) else data["documents"])]
    print(json.dumps(make_reference_snapshot(records,args.key), ensure_ascii=False, indent=2))
