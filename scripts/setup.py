#!/usr/bin/env python3
"""Install pinned model code into the currently active virtual environment."""

import argparse
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run(list(map(str, args)), check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "backbone",
        choices=(
            "flux1-kontext-dev",
            "flux2-klein-base-9b",
            "qwen-image-2.1",
            "sensenova-u1.5-8b-mot",
        ),
    )
    a = p.parse_args()
    group = "qwen" if a.backbone == "qwen-image-2.1" else "flux"
    run(
        sys.executable, "-m", "pip", "install", "-r", ROOT / f"requirements/{group}.txt"
    )
    lock = yaml.safe_load((ROOT / "upstream.lock").read_text())
    key, folder = {
        "flux1-kontext-dev": ("flux", "flux"),
        "sensenova-u1.5-8b-mot": ("sensenova", "SenseNova-U1"),
    }.get(a.backbone, (None, None))
    if key:
        target = ROOT / "upstream" / folder
        if not target.exists():
            run("git", "clone", lock[key]["repo"], target)
            run("git", "-C", target, "checkout", lock[key]["revision"])
    run(sys.executable, "-m", "pip", "install", "-e", str(ROOT) + "[evaluation,test]")
    print("Setup complete. Download the model snapshot and training source data next.")


if __name__ == "__main__":
    main()
