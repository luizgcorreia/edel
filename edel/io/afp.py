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


def load_afp_authors(afp_root: Path) -> dict[str, dict[str, Any]]:
    """Load the author lookup table from metadata/authors.toml."""
    authors_file = afp_root / "metadata" / "authors.toml"
    if not authors_file.exists():
        return {}

    with open(authors_file, "rb") as f:
        data = tomllib.load(f)

    author_map = {}
    for auth_id, info in data.items():
        if isinstance(info, dict):
            author_map[auth_id] = {
                "name": info.get("name", auth_id),
                "orcid": info.get("orcid"),
                "homepages": list(info.get("homepages", {}).values()) if isinstance(info.get("homepages"), dict) else []
            }
    return author_map


def load_afp_metadata(
    afp_root: Path, author_map: dict[str, dict[str, Any]]
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

        # Resolve authors into an OpenAlex-compatible schema
        entry_authors = []
        author_data = data.get("authors", {})
        if isinstance(author_data, dict):
            auth_ids = list(author_data.keys())
            for i, auth_id in enumerate(auth_ids):
                # Determine position
                position = "middle"
                if i == 0:
                    position = "first"
                elif i == len(auth_ids) - 1 and len(auth_ids) > 1:
                    position = "last"
                
                auth_info = author_map.get(auth_id, {"name": auth_id, "orcid": None})
                
                entry_authors.append({
                    "author": {
                        "id": f"afp:author:{auth_id}",
                        "display_name": auth_info["name"],
                        "orcid": auth_info.get("orcid")
                    },
                    "author_position": position,
                    "institutions": [],
                    "is_corresponding": i == 0 # Assumption: first author is corresponding
                })

        metadata[entry_id] = {
            "title": data.get("title", entry_id),
            "abstract": data.get("abstract", ""),
            "date": str(data.get("date", "")),
            "topics": data.get("topics", []),
            "authorships": entry_authors,
        }
    return metadata


def parse_afp_root(root_file: Path) -> tuple[list[str], list[str], list[str]]:
    """Parse an Isabelle ROOT file for session names, imports, and theories."""
    if not root_file.exists():
        return [], [], []

    text = root_file.read_text(errors="ignore")

    # Strip comments (* ... *)
    while True:
        new_text = re.sub(r'\(\*(?:[^*]|\*(?!\)))*\*\)', '', text, flags=re.DOTALL)
        if new_text == text:
            break
        text = new_text

    # Find all session blocks
    session_starts = [m.start() for m in re.finditer(r'\bsession\b', text)]
    session_blocks = []
    for i, start_idx in enumerate(session_starts):
        end_idx = session_starts[i+1] if i+1 < len(session_starts) else len(text)
        session_blocks.append(text[start_idx:end_idx])

    session_names = []
    all_imports = []
    all_theories = []

    keywords = ['options', 'sessions', 'directories', 'theories', 'document_files', 'document_theories']

    for block in session_blocks:
        if '=' not in block:
            continue
        
        header, rest = block.split('=', 1)
        
        session_name_match = re.search(r'\bsession\s+([A-Za-z0-9_\-\"\']+)', header)
        if not session_name_match:
            continue
        session_name = session_name_match.group(1).strip('"\'')
        session_names.append(session_name)
        
        keyword_pattern = r'\b(?:' + '|'.join(keywords) + r')\b'
        base_sessions_part = re.split(keyword_pattern, rest, maxsplit=1)[0]
        
        for part in base_sessions_part.split('+'):
            dep = part.strip().strip('"\'')
            if dep and dep not in all_imports:
                all_imports.append(dep)
                
        sessions_matches = list(re.finditer(r'\bsessions\b', rest))
        for m in sessions_matches:
            sub_rest = rest[m.end():]
            sessions_part = re.split(keyword_pattern, sub_rest, maxsplit=1)[0]
            tokens = re.findall(r'[A-Za-z0-9_\-\"\']+', sessions_part)
            for tok in tokens:
                dep = tok.strip('"\'')
                if dep and dep not in all_imports:
                    all_imports.append(dep)
                    
        theories_matches = list(re.finditer(r'\btheories\b', rest))
        for m in theories_matches:
            sub_rest = rest[m.end():]
            theories_part = re.split(keyword_pattern, sub_rest, maxsplit=1)[0]
            tokens = re.findall(r'[A-Za-z0-9_\-\.\/\"\'\\]+', theories_part)
            for tok in tokens:
                thy = tok.strip('"\'')
                if thy and thy not in all_theories:
                    all_theories.append(thy)

    return session_names, all_imports, all_theories


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
