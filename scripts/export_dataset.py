#!/usr/bin/env python3
"""Package source instructions or the frozen Add & Preserve benchmark for the Hub."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def asset(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Asset must be relative to the dataset root: {name}")
    return path


def copy_asset(pair):
    source, destination = pair
    digest = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or sha256(destination) != digest:
            raise ValueError(f"Existing release asset differs: {destination}")
    else:
        shutil.copyfile(source, destination)
        if sha256(destination) != digest:
            raise ValueError(f"Copy failed verification: {destination}")
    with Image.open(destination) as image:
        image.verify()
    return digest


def export_source(args):
    rows, images, contracts = [], {}, []
    provenance = {r["source_id"]: r for r in read_rows(args.provenance)}
    for raw in read_rows(args.input):
        sid = raw.get("source_id", raw.get("source_identity"))
        if not sid:
            raise ValueError(
                "Source identity is required to preserve the training split"
            )
        source = Path(raw.get("image", raw.get("source")))
        if not source.is_absolute():
            source = args.image_root / source
        task = {"local_edit": "local", "noop": "identity"}.get(raw["task"], raw["task"])
        if task not in ("local", "identity"):
            raise ValueError(f"Unknown source task: {task}")
        relative = str(
            Path("images")
            / sid.split(":", 1)[0]
            / hashlib.sha256(sid.encode()).hexdigest()[:2]
            / (re.sub(r"[^A-Za-z0-9_.-]", "_", sid) + source.suffix.lower())
        )
        if sid in images and images[sid][0] != source:
            raise ValueError(f"Source identity points to different images: {sid}")
        images[sid] = (source, relative)
        row = {
            "id": raw["id"],
            "source_id": sid,
            "image": relative,
            "prompt": raw["prompt"],
            "task": task,
            "seed": int(raw["seed"]),
        }
        for key in (
            "sampling_category",
            "source_ordinal",
            "prompt_id",
            "edit_type",
            "target",
            "category",
            "origin",
            "source_group",
        ):
            if key in raw:
                row[key] = raw[key]
        rows.append(row)
        contracts.append({k: v for k, v in row.items() if k != "image"})
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate trajectory request IDs")
    pairs = [(source, args.output / relative) for source, relative in images.values()]
    with ThreadPoolExecutor(args.workers) as pool:
        hashes = list(pool.map(copy_asset, pairs))
    public_provenance = []
    for (sid, (_, relative)), digest in zip(images.items(), hashes, strict=True):
        public_provenance.append(
            {**provenance[sid], "image": relative, "sha256": digest}
        )
    write_rows(args.output / "sources.jsonl", rows)
    write_rows(args.output / "provenance.jsonl", public_provenance)
    return {
        "name": "TailSteer-Source",
        "version": "1.0",
        "images": len(images),
        "trajectory_requests": len(rows),
        "tasks": dict(Counter(r["task"] for r in rows)),
        "categories": dict(Counter(r["category"] for r in public_provenance)),
        "instruction_contract_sha256": hashlib.sha256(
            json.dumps(contracts, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
    }


def export_benchmark(args):
    original = json.loads((args.input / "manifest.json").read_text())
    tasks = json.loads((args.input / "tasks.json").read_text())["tasks"]
    by_id = {t["id"]: t for t in tasks}
    provenance = {r["id"]: r for r in read_rows(args.provenance)}
    items, pairs, public_provenance = [], [], []
    fields = (
        "id",
        "category",
        "size",
        "bbox",
        "keep_fraction",
        "rounds",
        "original_size",
        "crop_box",
    )
    for raw in original["items"]:
        row = {k: raw[k] for k in fields if k in raw}
        expected = [{"prompt": r["prompt"], "seed": r["seed"]} for r in row["rounds"]]
        task = by_id[row["id"]]
        if (
            len(expected) != 10
            or task["rounds"] != expected
            or task["image"] != raw["image"]
        ):
            raise ValueError(f"Frozen instructions differ: {row['id']}")
        for key in ("image", "keep_mask", "original"):
            if key not in raw:
                continue
            source = asset(args.input, raw[key])
            if sha256(source) != raw[f"{key}_sha256"]:
                raise ValueError(f"Frozen asset changed: {raw[key]}")
            row[key] = raw[key]
            row[f"{key}_sha256"] = raw[f"{key}_sha256"]
            pairs.append((source, asset(args.output, raw[key])))
        items.append(row)
        public_provenance.append({**provenance[row["id"]], "image": row["image"]})
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(copy_asset, pairs))
    shutil.copyfile(args.input / "tasks.json", args.output / "tasks.json")
    manifest = {
        k: original[k]
        for k in (
            "schema_version",
            "prompt_language",
            "seed",
            "rounds",
            "preprocessing",
            "source_decoration",
            "prompt_template",
        )
    }
    manifest.update(
        name="add-preserve",
        version="1.0",
        items=items,
        tasks_sha256=sha256(args.output / "tasks.json"),
    )
    write_json(args.output / "manifest.json", manifest)
    write_rows(args.output / "provenance.jsonl", public_provenance)
    return {
        "name": "Add & Preserve",
        "version": "1.0",
        "images": len(items),
        "rounds_per_image": 10,
        "edit_requests": sum(len(r["rounds"]) for r in items),
        "categories": dict(Counter(r["category"] for r in items)),
        "tasks_sha256": manifest["tasks_sha256"],
    }


def validate(folder):
    for line in (folder / "checksums.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if sha256(asset(folder, name)) != expected:
            raise ValueError(f"Release checksum mismatch: {name}")
    if (folder / "sources.jsonl").exists():
        rows = read_rows(folder / "sources.jsonl")
        if len(rows) != len({r["id"] for r in rows}):
            raise ValueError("Duplicate request IDs")
        for row in rows:
            if not asset(folder, row["image"]).is_file():
                raise FileNotFoundError(row["image"])
    else:
        import numpy as np

        manifest = json.loads((folder / "manifest.json").read_text())
        tasks = json.loads((folder / "tasks.json").read_text())["tasks"]
        by_id = {r["id"]: r for r in manifest["items"]}
        if len(tasks) != len(by_id) or len(tasks) != len({t["id"] for t in tasks}):
            raise ValueError("Benchmark task IDs differ")
        for task in tasks:
            row = by_id[task["id"]]
            if task["image"] != row["image"] or task["rounds"] != [
                {"prompt": r["prompt"], "seed": r["seed"]} for r in row["rounds"]
            ]:
                raise ValueError(f"Benchmark prompts differ: {task['id']}")
            with Image.open(asset(folder, row["image"])) as image:
                if list(image.size) != row["size"]:
                    raise ValueError("Benchmark size changed")
            with Image.open(asset(folder, row["keep_mask"])) as mask:
                keep = np.asarray(mask.convert("L")) > 0
            width, height = row["size"]
            expected = np.ones((height, width), dtype=bool)
            left, top, right, bottom = row["bbox"]
            expected[top:bottom, left:right] = False
            if not np.array_equal(keep, expected):
                raise ValueError(f"Benchmark mask does not match bbox: {row['id']}")
    return json.loads((folder / "release.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)
    for kind in ("training-source", "add-preserve"):
        p = sub.add_parser(kind)
        p.add_argument("--input", type=Path, required=True)
        p.add_argument("--provenance", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--image-root", type=Path, default=Path("."))
        p.add_argument("--workers", type=int, default=8)
    sub.add_parser("validate").add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.kind == "validate":
        print(json.dumps(validate(args.folder), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    summary = (
        export_source(args)
        if args.kind == "training-source"
        else export_benchmark(args)
    )
    write_json(args.output / "release.json", summary)
    cards = ROOT / "docs/dataset-cards"
    shutil.copyfile(cards / f"{args.kind}.md", args.output / "README.md")
    shutil.copyfile(cards / "LICENSE.md", args.output / "LICENSE.md")
    shutil.copyfile(ROOT / "LICENSE", args.output / "LICENSE-APACHE-2.0.txt")
    files = sorted(
        p
        for p in args.output.rglob("*")
        if p.is_file()
        and p.name != "checksums.sha256"
        and not any(x.startswith(".") for x in p.relative_to(args.output).parts)
    )
    (args.output / "checksums.sha256").write_text(
        "".join(f"{sha256(p)}  {p.relative_to(args.output)}\n" for p in files)
    )
    print(json.dumps(validate(args.output), indent=2))


if __name__ == "__main__":
    main()
