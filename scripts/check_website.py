#!/usr/bin/env python3
"""Check the static project page and the documentation's shared assets."""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "website"
ASSET = re.compile(r"assets/[\w./-]+\.(?:webp|png|jpe?g|svg|ttf|woff2?|json|pdf|gif|txt)")
LOCAL_IMAGE = re.compile(r"^[\w./-]+\.(?:webp|png|jpe?g|svg|gif|json)(?:\?[^\s]*)?$")
PRIVATE = re.compile(r"/mnt/|/home/|/Users/|localhost|127\.0\.0\.1|file://|design/asset-provenance")


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


def main():
    errors, used, hashes = [], set(), {}

    def require(path, owner):
        path = path.resolve()
        if not path.is_relative_to(REPO) or not path.is_file():
            errors.append(f"{owner}: missing or out-of-repository file: {path}")
        else:
            used.add(path)

    for path in SITE.rglob("*"):
        if path.is_symlink():
            errors.append(f"Symlinks cannot be published: {path.relative_to(REPO)}")
        if not path.is_file():
            continue
        owner = path.relative_to(REPO)
        if path.suffix in {".py", ".log", ".pid", ".map"} or path.name == "provenance.json":
            errors.append(f"Non-public development file: {owner}")
        if path.suffix in {".html", ".css", ".js", ".json", ".svg", ".md"}:
            text = path.read_text()
            if PRIVATE.search(text):
                errors.append(f"Internal path or local preview reference: {owner}")
            if path.suffix in {".html", ".css", ".js"}:
                for asset in ASSET.findall(text):
                    require(SITE / asset, owner)
                for link in re.findall(r'(?:src|href)=["\']([^"\']+)["\']', text):
                    url = urlsplit(link)
                    if url.scheme or url.netloc or not url.path:
                        continue
                    if url.path.startswith("/"):
                        errors.append(f"Root-relative link breaks project Pages: {owner}: {link}")
                    else:
                        require(path.parent / unquote(url.path), owner)
            elif path.suffix == ".json":
                for value in strings(json.loads(text)):
                    if not LOCAL_IMAGE.fullmatch(value):
                        continue
                    base = SITE if value.startswith("assets/") else path.parent
                    require(base / urlsplit(value).path, owner)
        if path.suffix in {".webp", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ttf", ".woff2", ".pdf"}:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in hashes:
                errors.append(f"Duplicate asset: {owner} and {hashes[digest]}")
            hashes[digest] = owner

    # These names are chosen dynamically by the endpoint-composition animation.
    for name in ("curation-source", "curation-generated", "curation-keep_mask", "identity-source", "identity-generated"):
        require(SITE / "assets" / f"{name}.webp", "endpoint composition")

    documents = [REPO / "README.md", REPO / "THIRD_PARTY.md", *REPO.glob("docs/model-cards/*.md"), *SITE.rglob("*.md")]
    for path in documents:
        text = path.read_text()
        links = re.findall(r'\]\(([^\s)]+)\)|(?:src|href)=["\']([^"\']+)["\']', text)
        for markdown, html in links:
            link = urlsplit(markdown or html)
            if not link.scheme and not link.netloc and link.path and not (path.parent / link.path).is_dir():
                require(path.parent / unquote(link.path), path.relative_to(REPO))

    assets = {path.resolve() for path in (SITE / "assets").rglob("*") if path.is_file()}
    # Manifests, license notices and asset documentation are kept alongside images.
    unused = {path for path in assets - used if path.suffix not in {".json", ".txt", ".md"}}
    errors.extend(f"Unreferenced asset: {path.relative_to(REPO)}" for path in sorted(unused))

    sequence = json.loads((SITE / "assets/frequency-sequence.json").read_text())
    mse = json.loads((SITE / "assets/frequency-mse.json").read_text())
    if [frame["step"] for frame in sequence["frames"]] != list(range(51)):
        errors.append("Frequency animation must contain every denoising step from 0 to 50")
    if sequence["data_sha256"] != mse["sequence_data_sha256"]:
        errors.append("Frequency and preservation-error measurements must refer to the same sequence")

    if errors:
        raise SystemExit("Website checks failed:\n" + "\n".join(errors))
    size = sum(path.stat().st_size for path in assets) / 1024**2
    print(f"Website checks passed: {len(assets)} assets ({size:.1f} MiB); shared documentation links valid; 51 trajectory states.")


if __name__ == "__main__":
    main()
