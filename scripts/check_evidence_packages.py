"""Reject measured archives and paper/evidence copies in built distributions."""

import tarfile
import zipfile
from pathlib import Path


def check(directory: Path) -> None:
    packages = list(directory.glob("*.tar.gz")) + list(directory.glob("*.whl"))
    if not packages:
        raise ValueError("No built distributions")
    for package in packages:
        if package.suffix == ".whl":
            with zipfile.ZipFile(package) as archive:
                names = archive.namelist()
        else:
            with tarfile.open(package) as archive:
                names = archive.getnames()
        for name in names:
            parts = Path(name).parts
            if "results" in parts or "paper" in parts or "docs/evidence/" in name:
                raise ValueError(f"Evidence bundled in {package}: {name}")
    print(f"Checked {len(packages)} distributions: no bundled evidence")


if __name__ == "__main__":
    check(Path("dist"))
