"""Parser for AFP entry metadata TOML files."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any


def find_afp_metadata_dir(search_path: str | Path = ".") -> Path | None:
    """Locate the AFP metadata directory by walking/searching typical paths."""
    search_path = Path(search_path).resolve()
    
    # Try typical paths
    candidates = [
        search_path / "external" / "afp-2025-2" / "metadata" / "entries",
        search_path / "external" / "afp" / "metadata" / "entries",
        search_path / "metadata" / "entries",
    ]
    
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
            
    # Recursive search as fallback
    for p in search_path.glob("**/metadata/entries"):
        if p.is_dir():
            return p
            
    # Try parent directory in case we are in a subfolder
    if search_path.parent != search_path:
        return find_afp_metadata_dir(search_path.parent)
        
    return None


class AFPMetadataParser:
    """Loader and parser for AFP metadata files."""

    def __init__(self, metadata_dir: str | Path | None = None):
        if metadata_dir:
            self.metadata_dir = Path(metadata_dir).resolve()
        else:
            self.metadata_dir = find_afp_metadata_dir()
            
        if self.metadata_dir:
            print(f"AFP Metadata Parser initialized using directory: {self.metadata_dir}")
        else:
            print("Warning: AFP metadata directory not found. Metadata enrichment will be skipped.")

    def load_entry_metadata(self, entry_name: str) -> dict[str, Any]:
        """Load and return metadata for a specific AFP entry.
        
        Returns an empty dict if the entry cannot be found or parsed.
        """
        if not self.metadata_dir:
            return {}
            
        # Standardize entry_name (replace underscores with hyphens for TOML filenames)
        name_variants = [
            entry_name,
            entry_name.replace("_", "-"),
            entry_name.replace("-", "_")
        ]
        
        toml_path = None
        for variant in name_variants:
            candidate = self.metadata_dir / f"{variant}.toml"
            if candidate.exists():
                toml_path = candidate
                break
                
        if not toml_path:
            # Flexible case-insensitive and dash/underscore-agnostic matching
            def normalize(s: str) -> str:
                return s.lower().replace("_", "").replace("-", "")
                
            norm_name = normalize(entry_name)
            for file in self.metadata_dir.glob("*.toml"):
                if normalize(file.stem) == norm_name:
                    toml_path = file
                    break
                    
        if not toml_path or not toml_path.exists():
            return {}
            
        try:
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
            return {
                "title": data.get("title", ""),
                "date": str(data.get("date", "")),
                "topics": data.get("topics", []),
                "abstract": data.get("abstract", "").strip(),
                "license": data.get("license", ""),
                "authors": list(data.get("authors", {}).keys()),
            }
        except Exception as e:
            print(f"Error reading AFP metadata for {entry_name}: {e}")
            return {}
