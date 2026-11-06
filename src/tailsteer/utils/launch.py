"""Distributed launch without requiring a duplicated Torch install in venvs."""

from pathlib import Path


def torchrun_command(executable: str | Path) -> list[str]:
    path = Path(executable)
    python = path.with_name("python")
    # A system-site-packages venv can import Torch but has no bin/torchrun.
    if (
        path.name == "torchrun"
        and path.is_absolute()
        and not path.is_file()
        and python.is_file()
    ):
        return [str(python), "-m", "torch.distributed.run"]
    return [str(executable)]
