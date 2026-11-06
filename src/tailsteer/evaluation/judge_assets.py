"""EdiVal assets: official downloads or local snapshots, same native loaders."""

import shutil
import urllib.request
from pathlib import Path

from .common import atomic_text, digest, file_digest, read_json, write_json

# Keys are the identifiers used verbatim by the pinned official evaluators.
MODELS = {
    "Qwen/Qwen2-VL-7B-Instruct": "Qwen2-VL-7B-Instruct",
    "facebook/dinov3-vitb16-pretrain-lvd1689m": "dinov3-vitb16-pretrain-lvd1689m",
    "google/vit-large-patch16-384": "vit-large-patch16-384",
    "t5-base": "t5-base",
    "bert-base-uncased": "bert-base-uncased",
    "C-Tianyu/RAHF": "RAHF",
}
ALIASES = {
    "t5-base": "google-t5/t5-base",
    "bert-base-uncased": "google-bert/bert-base-uncased",
}
GROUNDING_FILE = "groundingdino_swint_ogc.pth"


def _model_files(source: Path, repo: str) -> list[Path]:
    """Select native PyTorch assets, excluding TF/Flax/CoreML duplicates."""
    if repo == "C-Tianyu/RAHF":
        files = [source / "rahf_model.pt"]
    else:
        required = ["config.json"]
        if repo in {"Qwen/Qwen2-VL-7B-Instruct", "t5-base", "bert-base-uncased"}:
            required += {
                "Qwen/Qwen2-VL-7B-Instruct": [
                    "tokenizer_config.json",
                    "tokenizer.json",
                ],
                "t5-base": ["spiece.model"],
                "bert-base-uncased": ["vocab.txt"],
            }[repo]
        if repo in {
            "Qwen/Qwen2-VL-7B-Instruct",
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "google/vit-large-patch16-384",
        }:
            required.append("preprocessor_config.json")
        for name in required:
            if not (source / name).is_file():
                raise FileNotFoundError(source / name)
        files = [
            p
            for p in source.iterdir()
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix in {".json", ".txt", ".model"}
            and not p.name.endswith(".index.json")
        ]
        for name in (
            "model.safetensors.index.json",
            "model.safetensors",
            "pytorch_model.bin.index.json",
            "pytorch_model.bin",
        ):
            weight = source / name
            if not weight.is_file():
                continue
            files.append(weight)
            if name.endswith(".index.json"):
                mapping = read_json(weight).get("weight_map")
                if not isinstance(mapping, dict) or not mapping:
                    raise ValueError(f"Invalid weight index: {weight}")
                for shard in sorted(set(mapping.values())):
                    relative = Path(shard)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"Unsafe shard path: {shard}")
                    files.append(source / relative)
            break
        else:
            raise FileNotFoundError(f"No PyTorch weights in {source}")
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty asset: {path}")
    return sorted(set(files))


def _file_record(path: Path) -> dict:
    before = path.stat()
    sha = file_digest(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Asset changed during registration: {path}")
    return {"sha256": sha, "size": after.st_size, "mtime_ns": after.st_mtime_ns}


def _snapshot(root: Path, repo: str, revision: str) -> Path:
    return (
        root
        / "cache/huggingface/hub"
        / ("models--" + repo.replace("/", "--"))
        / "snapshots"
        / revision
    )


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            raise FileExistsError(f"Refusing to replace existing asset: {target}")
    else:
        target.symlink_to(source.resolve())


def register_local_edival_judges(root: Path, directory: Path) -> dict:
    """Register available assets offline; missing models remain explicitly unready."""
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    path = root / "models/edival-models.json"
    manifest = read_json(path) if path.exists() else {}
    for repo, name in MODELS.items():
        source = directory / name
        if not source.is_dir():
            continue
        print(f"Registering EdiVal asset: {repo} <- {source}", flush=True)
        files = {
            p.relative_to(source).as_posix(): _file_record(p)
            for p in _model_files(source, repo)
        }
        # Content identity, not a fabricated Hugging Face upstream commit.
        revision = digest({k: v["sha256"] for k, v in files.items()})[:40]
        aliases = [repo] + ([ALIASES[repo]] if repo in ALIASES else [])
        for alias in aliases:
            snapshot = _snapshot(root, alias, revision)
            for name in files:
                _link(source / name, snapshot / name)
            atomic_text(snapshot.parent.parent / "refs/main", revision)
        receipt = source / ".tailsteer-download.json"
        origin = read_json(receipt) if receipt.exists() else {}
        manifest[repo] = {
            "revision": revision,
            "revision_kind": "local-content-sha256",
            "path": str(_snapshot(root, repo, revision)),
            "source_path": str(source),
            "source_provider": origin.get("provider", "local"),
            "source_model_id": origin.get("source_model_id", origin.get("model_id")),
            "aliases": aliases,
            "files": files,
        }
        write_json(path, manifest)
    grounding = directory / "groundingdino" / GROUNDING_FILE
    if grounding.is_file():
        target = root / "models/groundingdino" / GROUNDING_FILE
        _link(grounding, target)
        write_json(target.parent / "source.json", _file_record(target))
    missing = missing_edival_judges(root, manifest)
    if not missing:
        verify_edival_judges(root)
    return {"registered": list(manifest), "missing": missing, "ready": not missing}


def missing_edival_judges(root: Path, manifest: dict) -> list[str]:
    missing = [
        repo
        for repo in MODELS
        if not manifest.get(repo, {}).get("path")
        or not Path(manifest[repo]["path"]).is_dir()
    ]
    if not (root / "models/groundingdino" / GROUNDING_FILE).is_file():
        missing.append(GROUNDING_FILE)
    if not (root / "models/groundingdino/source.json").is_file():
        missing.append("groundingdino/source.json")
    return missing


def verify_edival_judges(root: Path, *, hashes: bool = False) -> dict:
    path = root / "models/edival-models.json"
    manifest = read_json(path) if path.exists() else {}
    missing = missing_edival_judges(root, manifest)
    if missing:
        raise FileNotFoundError("EdiVal assets not ready: " + ", ".join(missing))
    for repo, asset in manifest.items():
        snapshot = Path(asset["path"])
        for alias in asset.get("aliases", [repo]):
            candidate = (
                _snapshot(root, alias, asset["revision"]) if alias != repo else snapshot
            )
            ref = candidate.parent.parent / "refs/main"
            if not ref.is_file() or ref.read_text().strip() != asset["revision"]:
                raise ValueError(f"EdiVal cache reference changed: {ref}")
            for name, record in asset.get("files", {}).items():
                _verify_file(candidate / name, record, hashes=hashes)
    grounding = root / "models/groundingdino"
    _verify_file(
        grounding / GROUNDING_FILE, read_json(grounding / "source.json"), hashes=hashes
    )
    return manifest


def _verify_file(path: Path, record: dict, *, hashes: bool) -> None:
    stat = path.stat()
    if stat.st_size == 0 or any(
        key in record and record[key] != actual
        for key, actual in (("size", stat.st_size), ("mtime_ns", stat.st_mtime_ns))
    ):
        raise ValueError(f"EdiVal asset changed; register again: {path}")
    if hashes and file_digest(path) != record["sha256"]:
        raise ValueError(f"EdiVal asset checksum mismatch: {path}")


def prepare_edival_judges(root: Path):
    from huggingface_hub import HfApi, snapshot_download

    path = root / "models/edival-models.json"
    manifest = read_json(path) if path.exists() else {}
    for repo in MODELS:
        if manifest.get(repo, {}).get("revision_kind") == "local-content-sha256":
            print(f"Reusing registered local EdiVal asset: {repo}", flush=True)
            continue
        if repo not in manifest:
            print(f"Resolving EdiVal asset revision: {repo}", flush=True)
            manifest[repo] = {"revision": HfApi().model_info(repo, timeout=30).sha}
            write_json(path, manifest)
        revision = manifest[repo]["revision"]
        print(f"EdiVal judge: {repo}@{revision}", flush=True)
        location = snapshot_download(repo, revision=revision)
        # Official loaders use default revision=main. Pin that cache reference;
        # evaluation runs offline and does not mutate the official loader code.
        atomic_text(Path(location).parent.parent / "refs/main", revision)
        manifest[repo]["path"] = location
        write_json(path, manifest)
    weights = root / "models/groundingdino/groundingdino_swint_ogc.pth"
    weights.parent.mkdir(parents=True, exist_ok=True)
    if not weights.exists():
        url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        partial = weights.with_suffix(".partial")
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            partial.open("wb") as f,
        ):
            shutil.copyfileobj(response, f)
        partial.replace(weights)
    write_json(root / "models/groundingdino/source.json", _file_record(weights))
    verify_edival_judges(root)
