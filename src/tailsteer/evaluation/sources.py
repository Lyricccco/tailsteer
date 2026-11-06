"""Download versioned official assets without executing upstream setup scripts."""

from __future__ import annotations

import fnmatch
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from .common import contained, file_digest, load_sources, read_json, write_json


def _safe_member(name: str) -> bool:
    p = PurePosixPath(name)
    return not p.is_absolute() and ".." not in p.parts


def extract_archive(
    archive: Path, output: Path, *, strip_first=False, include=None
) -> None:
    """Extract regular files only; archives cannot create escaping paths or links."""

    def destination(name):
        if not _safe_member(name):
            raise ValueError(f"Unsafe archive member: {name}")
        if strip_first:
            name = name.partition("/")[2]
        if not name or (include is not None and not include(name)):
            return None
        target = contained(output, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                if item.is_dir():
                    continue
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(
                        f"Archive symlink is not supported: {item.filename}"
                    )
                target = destination(item.filename)
                if target:
                    with z.open(item) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(archive) as tar:
            for item in tar:
                if not item.isfile():
                    continue
                target = destination(item.name)
                if target:
                    with tar.extractfile(item) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)


def prepare_code(root: Path, benchmark: str, *, spec: dict | None = None) -> Path:
    spec = load_sources()[benchmark] if spec is None else spec
    destination = root / "upstream" / benchmark / spec["code_revision"]
    receipt = destination / "source.json"
    if not receipt.exists():
        paths = spec["code_paths"]

        def selected(name):
            return (
                any(
                    name == p or (p.endswith("/") and name.startswith(p)) for p in paths
                )
                and not any(
                    x in name.split("/") for x in (".git", "__pycache__", ".asset")
                )
                and not name.endswith(("secret.env", ".pyc"))
            )

        url = f"https://codeload.github.com/{spec['code_repo']}/tar.gz/{spec['code_revision']}"
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tailsteer-eval-") as tmp:
            archive = Path(tmp) / "source.tar.gz"
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                archive.open("wb") as f,
            ):
                shutil.copyfileobj(response, f)
            extract_archive(archive, destination, strip_first=True, include=selected)
        files = {
            str(p.relative_to(destination)): file_digest(p)
            for p in destination.rglob("*")
            if p.is_file()
        }
        write_json(
            receipt,
            {
                "repo": spec["code_repo"],
                "revision": spec["code_revision"],
                "files": files,
            },
        )
    return destination / spec.get("code_subdir", "")


def verify_code(root: Path, benchmark: str, *, spec: dict | None = None) -> Path:
    spec = load_sources()[benchmark] if spec is None else spec
    base = root / "upstream" / benchmark / spec["code_revision"]
    receipt = read_json(base / "source.json")
    for name, expected in receipt["files"].items():
        if file_digest(contained(base, name)) != expected:
            raise ValueError(f"Official source was modified: {base / name}")
    return base / spec.get("code_subdir", "")


def prepare_dataset(root: Path, benchmark: str) -> Path:
    from huggingface_hub import HfApi, hf_hub_download

    spec = load_sources()[benchmark]
    parent = root / "datasets" / benchmark
    selection = parent / "revision.json"
    # A mutable source is resolved once, never silently updated on resume.
    if selection.exists():
        revision = read_json(selection)["revision"]
    else:
        requested = spec["dataset_revision"]
        revision = (
            requested
            if len(requested) == 40
            else HfApi().dataset_info(spec["dataset_repo"], revision=requested).sha
        )
        write_json(selection, {"repo": spec["dataset_repo"], "revision": revision})
    destination = parent / revision
    receipt = destination / "source.json"
    if receipt.exists():
        return destination
    files = HfApi().list_repo_files(
        spec["dataset_repo"], repo_type="dataset", revision=revision
    )
    selected = [
        f
        for f in files
        if any(fnmatch.fnmatch(f, pattern) for pattern in spec["dataset_files"])
    ]
    if not selected:
        raise ValueError(f"No dataset assets matched for {benchmark}")
    hashes = {}
    for name in selected:
        print(f"Downloading {benchmark}: {name}", flush=True)
        path = Path(
            hf_hub_download(
                spec["dataset_repo"],
                filename=name,
                repo_type="dataset",
                revision=revision,
                local_dir=destination,
            )
        )
        hashes[name] = file_digest(path)
        if name.endswith((".zip", ".tar", ".tar.gz")):
            extract_archive(path, destination / "extracted")
    write_json(
        receipt, {"repo": spec["dataset_repo"], "revision": revision, "files": hashes}
    )
    return destination
