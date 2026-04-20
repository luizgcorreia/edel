"""AFP I/O helpers for repository management and parsing."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    # Fallback for Python < 3.11 if needed, but we are on 3.11
    import tomli as tomllib


def ensure_afp_repo(repo_url: str) -> Path:
    """Ensure the AFP repository is cloned and available."""
    # Use a directory name based on the last part of the URL
    repo_name = repo_url.split("/")[-1]
    base_path = Path(__file__).parents[2] / "external" / repo_name

    if not (base_path / ".hg").exists():
        print(f"Cloning AFP repository {repo_url} to {base_path}...")
        base_path.parent.mkdir(parents=True, exist_ok=True)
        # We need hg from the environment
        subprocess.run(["hg", "clone", repo_url, str(base_path)], check=True)
    return base_path


def load_afp_authors(afp_root: Path) -> dict[str, str]:
    """Load the author lookup table from metadata/authors.toml."""
    authors_file = afp_root / "metadata" / "authors.toml"
    if not authors_file.exists():
        return {}

    with open(authors_file, "rb") as f:
        data = tomllib.load(f)

    author_map = {}
    for auth_id, info in data.items():
        if isinstance(info, dict):
            author_map[auth_id] = info.get("name", auth_id)
    return author_map


def load_afp_metadata(
    afp_root: Path, author_map: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Load entry metadata from the metadata/entries directory."""
    entries_dir = afp_root / "metadata" / "entries"
    metadata = {}

    if not entries_dir.exists():
        return {}

    for toml_file in entries_dir.glob("*.toml"):
        entry_id = toml_file.stem
        with open(toml_file, "rb") as f:
            data = tomllib.load(f)

        # Resolve authors
        entry_authors = []
        author_data = data.get("authors", {})
        if isinstance(author_data, dict):
            for auth_id in author_data.keys():
                entry_authors.append(
                    {"id": auth_id, "display_name": author_map.get(auth_id, auth_id)}
                )

        metadata[entry_id] = {
            "title": data.get("title", entry_id),
            "abstract": data.get("abstract", ""),
            "date": str(data.get("date", "")),
            "topics": data.get("topics", []),
            "authorships": entry_authors,
        }
    return metadata


def parse_afp_root(root_file: Path) -> tuple[str | None, list[str], list[str]]:
    """Parse an Isabelle ROOT file for session name, imports, and theories."""
    if not root_file.exists():
        return None, [], []

    text = root_file.read_text(errors="ignore")

    # Session name
    s = re.search(r'session\s+"(.+?)"', text)
    session_name = s.group(1) if s else None

    # Imports (base sessions)
    imports = re.findall(r'"([^"]+)"', text)

    # Theories block
    tblock = re.search(r"theories(.+?)document_files", text, re.S)
    if not tblock:
        # fallback if no document_files
        tblock = re.search(r"theories(.+?)$", text, re.S)

    theories = []
    if tblock:
        theories = re.findall(r"[A-Za-z0-9_]+", tblock.group(1))

    return session_name, imports, theories


def parse_thy_entities(theory_dir: Path) -> tuple[list[str], list[str]]:
    """Extract definitions and lemmas from all .thy files in a directory."""
    definitions = []
    lemmas = []

    for thy_file in theory_dir.glob("*.thy"):
        with open(thy_file, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("definition"):
                    m = re.search(r"definition\s+([A-Za-z0-9_']+)", line)
                    if m:
                        definitions.append(m.group(1))
                elif line.startswith(("lemma", "theorem", "corollary")):
                    m = re.search(r"(?:lemma|theorem|corollary)\s+([A-Za-z0-9_']+)", line)
                    if m:
                        lemmas.append(m.group(1))
    return definitions, lemmas
