"""Translate official datasets into the existing multi-round task format."""

from __future__ import annotations

import ast
import csv
import shutil
import tempfile
from io import BytesIO
from pathlib import Path

from .common import contained, digest, file_digest, read_json, write_json


def find_one(root: Path, name: str) -> Path:
    direct = root / name
    if direct.is_file():
        return direct
    matches = sorted(p for p in root.rglob(name) if ".cache" not in p.parts)
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} below {root}, found {len(matches)}")
    return matches[0]


def _arrow_rows(root: Path):
    from datasets import DatasetDict, load_dataset, load_from_disk

    if (root / "state.json").exists() or (root / "dataset_dict.json").exists():
        dataset = load_from_disk(str(root))
        return dataset["train"] if isinstance(dataset, DatasetDict) else dataset
    parquet = sorted(str(p) for p in root.rglob("*.parquet"))
    if not parquet:
        raise FileNotFoundError(f"No Arrow dataset or parquet found in {root}")
    return load_dataset("parquet", data_files={"train": parquet}, split="train")


def official_rows(benchmark: str, raw: Path, code: Path):
    """Yield official identity, source, unchanged prompts and metadata."""
    if benchmark == "imgedit-basic":
        tasks = read_json(code / "basic_edit.json")
        images = find_one(raw, "singleturn.json").parent
        if read_json(images / "singleturn.json") != tasks:
            raise ValueError(
                "ImgEdit data metadata differs from pinned Basic benchmark"
            )
        for key, item in tasks.items():
            yield (
                key,
                contained(images, item["id"]),
                [item["prompt"]],
                {"key": key, **item},
            )
    elif benchmark == "rise":
        path = find_one(raw, "datav2_total_w_subtask.json")
        for item in read_json(path):
            image = path.parent / item["image"]
            if not image.is_file():
                image = find_one(raw, item["image"])
            yield item["index"], image, [item["instruction"]], item
    elif benchmark == "gedit":
        dataset = _arrow_rows(raw)
        if hasattr(dataset, "cast_column"):
            from datasets import Image as DatasetImage

            dataset = dataset.cast_column("input_image_raw", DatasetImage(decode=False))
        for item in dataset:
            meta = {
                k: v
                for k, v in item.items()
                if k not in {"input_image", "input_image_raw"}
            }
            identity = [item["task_type"], item["instruction_language"], item["key"]]
            yield identity, item["input_image_raw"], [item["instruction"]], meta
    elif benchmark == "mse":
        for item in _arrow_rows(raw):
            if len(item["context"]) != 5:
                raise ValueError("Pinned MSE protocol requires five turns")
            yield (
                item["index"],
                item["image"],
                item["context"],
                {k: v for k, v in item.items() if k != "image"},
            )
    elif benchmark == "edival":
        with (code / "oai_instruction_generation_output.csv").open() as f:
            rows = list(csv.DictReader(f))
        groups = {}
        for row in rows:
            groups.setdefault(int(row["image_index"]), []).append(row)
        for index, items in sorted(groups.items()):
            items.sort(key=lambda x: int(x["turns"]))
            if [int(x["turns"]) for x in items] != [1, 2, 3]:
                raise ValueError(
                    f"EdiVal image {index} does not have exactly three turns"
                )
            prompts = [ast.literal_eval(x["instructions"])[-1] for x in items]
            image = list(raw.rglob(f"{index}_input_raw.*"))
            image = [p for p in image if p.suffix.lower() in {".jpg", ".png", ".jpeg"}]
            if len(image) != 1:
                raise ValueError(f"Missing/ambiguous EdiVal source {index}: {image}")
            yield index, image[0], prompts, {"index": index, "rows": items}
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def prepare_benchmark(root: Path, benchmark: str, raw: Path, code: Path) -> Path:
    from PIL import Image

    output = root / "prepared" / benchmark
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for identity, source, prompts, metadata in official_rows(benchmark, raw, code):
        task_id = f"{benchmark}-{digest(identity)[:20]}"
        encoded = None
        if isinstance(source, dict):
            encoded = source.get("bytes")
            source = Image.open(BytesIO(encoded)) if encoded else Path(source["path"])
        suffix = (
            source.suffix.lower()
            if isinstance(source, Path)
            else (".jpg" if encoded and source.format == "JPEG" else ".png")
        )
        relative = f"images/{task_id}{suffix}"
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(source) if isinstance(source, Path) else source
        image = image.convert("RGB")
        valid = False
        if target.exists():
            try:
                with Image.open(target) as previous:
                    if (
                        previous.size != image.size
                        or previous.convert("RGB").tobytes() != image.tobytes()
                    ):
                        raise ValueError(f"Prepared image has changed: {target}")
                valid = True
            except OSError:
                if (output / "manifest.json").exists():
                    raise
                # Only an unfinished preparation may repair an interrupted copy.
        if not valid:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, suffix=suffix, delete=False
            ) as f:
                temporary = Path(f.name)
                if isinstance(source, Path):
                    with source.open("rb") as original:
                        shutil.copyfileobj(original, f)
                elif encoded:
                    f.write(encoded)
                else:
                    image.save(f, format="PNG")
            temporary.replace(target)
        # RISE's reference image is sometimes different from its generation input.
        if benchmark == "rise" and metadata.get("reference_img"):
            ref = find_one(raw, metadata["reference_img"])
            ref_target = contained(output / "references", metadata["reference_img"])
            ref_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ref, ref_target)
            metadata = {**metadata, "_reference_sha256": file_digest(ref_target)}
        records.append(
            {
                "id": task_id,
                "official_id": identity,
                "image": relative,
                "image_sha256": file_digest(target),
                "prompts": list(prompts),
                "metadata": metadata,
            }
        )
        if len(records) % 100 == 0:
            print(f"{benchmark}: prepared {len(records)} sessions", flush=True)
    if not records or len({x["id"] for x in records}) != len(records):
        raise ValueError("Benchmark must have nonempty unique identities")
    manifest = {
        "schema_version": 1,
        "benchmark": benchmark,
        "records": records,
        "raw_source": str(raw.resolve()),
        "official_code": str(code.resolve()),
        "sessions": len(records),
        "turns": sum(len(r["prompts"]) for r in records),
    }
    manifest["fingerprint"] = digest(records)
    path = output / "manifest.json"
    if path.exists() and read_json(path)["fingerprint"] != manifest["fingerprint"]:
        raise ValueError(
            f"Prepared dataset changed; use a different --eval-root: {path}"
        )
    write_json(path, manifest)
    return path


def verify_prepared(path: Path) -> dict:
    manifest = read_json(path)
    if digest(manifest["records"]) != manifest["fingerprint"]:
        raise ValueError(f"Prepared metadata checksum mismatch: {path}")
    for row in manifest["records"]:
        if manifest.get("task_type") == "t2i":
            if not row["prompts"] or not row["prompts"][0].strip():
                raise ValueError("Empty T2I prompt")
            continue
        if file_digest(contained(path.parent, row["image"])) != row["image_sha256"]:
            raise ValueError(f"Prepared source checksum mismatch: {row['id']}")
        if "_reference_sha256" in row["metadata"]:
            meta = row["metadata"]
            ref = contained(path.parent / "references", meta["reference_img"])
            if file_digest(ref) != meta["_reference_sha256"]:
                raise ValueError(f"Reference checksum mismatch: {row['id']}")
    return manifest
