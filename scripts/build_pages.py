#!/usr/bin/env python3
"""Build the dependency-free OP-Bench demo for GitHub Pages.

The local demo is served by web/backend/server.py. GitHub Pages only publishes
static files, so this script places the frontend and its read-only JSON
payloads under one directory with index.html at the root.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


STATIC_EXTENSIONS = {".css", ".html", ".js"}


def build_site(project_root: Path, output_dir: Path) -> None:
    """Copy the frontend and demo data into a Pages-ready directory."""

    frontend_dir = project_root / "web" / "frontend"
    data_dir = project_root / "web" / "data"
    if not frontend_dir.is_dir():
        raise FileNotFoundError(f"Frontend directory not found: {frontend_dir}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    output_dir = output_dir.resolve()
    project_root = project_root.resolve()
    if output_dir == project_root or output_dir == project_root / "web":
        raise ValueError("Refusing to use the project or web directory as output.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory must be empty or absent: {output_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(frontend_dir.iterdir()):
        if source.is_file() and source.suffix in STATIC_EXTENSIONS:
            shutil.copy2(source, output_dir / source.name)

    output_data_dir = output_dir / "data"
    output_data_dir.mkdir()
    for source in sorted(data_dir.glob("*.json")):
        shutil.copy2(source, output_data_dir / source.name)

    # Prevent an implicit Jekyll build from changing the dependency-free site.
    (output_dir / ".nojekyll").touch()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=".pages",
        type=Path,
        help="Empty directory to receive the deployable site (default: .pages)",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = args.output if args.output.is_absolute() else Path.cwd() / args.output
    build_site(project_root, output_dir)
    print(f"Built GitHub Pages site at {output_dir.resolve()}")


if __name__ == "__main__":
    main()
