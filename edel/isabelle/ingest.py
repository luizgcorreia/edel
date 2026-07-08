"""Isabelle/AFP Ingestion script to construct structured datasets."""

from __future__ import annotations

import os
import re
import socket
import pandas as pd
from pathlib import Path
from typing import Any

from edel.isabelle.parser import parse_source_segments, group_segments_to_lemmas
from edel.isabelle.aspects import extract_aspects
from edel.isabelle.metadata import AFPMetadataParser

SENTINEL = "<<DONE>>"


class EphemeralReplClient:
    """A socket-based client to communicate with the running Isabelle/REPL server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9147, token: str = ""):
        self.host = host
        self.port = port
        self.token = token or os.getenv("IR_AUTH_TOKEN", "")

    def send(self, ml_command: str) -> str:
        """Send command to ML server and return output."""
        if not self.token:
            raise RuntimeError("Authentication token not set. Set IR_AUTH_TOKEN env var.")
            
        sock = socket.create_connection((self.host, self.port), timeout=30)
        try:
            # 1. Authenticate
            sock.sendall((self.token + "\n").encode())
            auth_buf = b""
            while b"\n" not in auth_buf:
                chunk = sock.recv(1024)
                if not chunk:
                    raise RuntimeError("Connection closed during auth handshake")
                auth_buf += chunk
            if not auth_buf.startswith(b"OK"):
                raise RuntimeError("REPL authentication failed")
                
            # 2. Send command
            cmd = ml_command.strip()
            if not cmd.endswith(";"):
                cmd += ";"
            sock.sendall((cmd + "\n").encode())
            
            # 3. Read until sentinel
            buf = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    raise EOFError("Connection closed by repl.py")
                buf += chunk
                text = buf.decode("utf-8", errors="replace")
                if SENTINEL in text:
                    raw = text[:text.index(SENTINEL)].strip()
                    # Strip control characters
                    return raw.replace("\x05", "").replace("\x06", "")
        finally:
            sock.close()



def ingest_session_lemmas(
    host: str = "127.0.0.1",
    port: int = 9147,
    token: str = "",
    theory_filter: str | None = None
) -> pd.DataFrame:
    """Connect to a running I/R instance and parse all theories into a dataframe.
    
    Args:
        host: I/R server host
        port: I/R server port
        token: I/R authentication token
        theory_filter: optional regex pattern to filter theories to ingest
    """
    client = EphemeralReplClient(host=host, port=port, token=token)
    metadata_parser = AFPMetadataParser()
    
    print("Configuring Isabelle REPL to output full command spans (disabling 80-char truncation)...")
    client.send("Ir.config (fn cfg => {color = #color cfg, show_ignored = #show_ignored cfg, full_spans = true, show_theory_in_source = #show_theory_in_source cfg, auto_replay = #auto_replay cfg});")
    
    print("Fetching loaded theories...")
    raw_thys = client.send("Ir.theories ();")
    theories = [t.strip() for t in raw_thys.splitlines() if t.strip()]
    print(f"Found {len(theories)} theories loaded in Isabelle session.")
    
    if theory_filter:
        pattern = re.compile(theory_filter)
        theories = [t for t in theories if pattern.search(t)]
        print(f"Filtered to {len(theories)} theories matching pattern: '{theory_filter}'")
        
    records = []
    
    for theory in theories:
        print(f"Processing theory: {theory}...")
        try:
            # 1. Fetch source segments
            raw_source = client.send(f'Ir.source "{theory}" 0 ~1;')
            if not raw_source.strip():
                print(f"  No source commands available for {theory}")
                continue
                
            segments = parse_source_segments(raw_source)
            
            # 2. Fetch source map
            raw_map = client.send(f'Ir.source_map "{theory}" 0 ~1;')
            seg_map = {}
            for line in raw_map.splitlines():
                m = re.match(r'\s*(\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\S+)', line)
                if m:
                    seg_map[int(m.group(1))] = {
                        "keyword": m.group(2),
                        "line": int(m.group(3)),
                        "offset": int(m.group(4)),
                        "file": m.group(5).strip(),
                        "theory": theory
                    }
                    
            if not seg_map:
                print(f"  No source map available for {theory}")
                continue
                
            # 3. Extract header for theory context
            theory_header = ""
            for idx in sorted(segments.keys()):
                if seg_map.get(idx, {}).get("keyword") == "theory":
                    theory_header = segments[idx]
                    break
                    
            # 4. Group segments into lemma units
            lemmas = group_segments_to_lemmas(seg_map, segments)
            print(f"  Found {len(lemmas)} lemmas/theorems.")
            
            # 5. Get entry metadata
            # For AFP theories, name is usually Entry_Name.Theory_Name
            entry_name = theory.split('.')[0] if '.' in theory else theory
            entry_meta = metadata_parser.load_entry_metadata(entry_name)
            
            # Parse publication year from metadata date (typically YYYY-MM-DD)
            date_str = entry_meta.get("date", "")
            pub_year = 2025  # Default/fallback to current Isabelle/AFP session year
            if date_str:
                try:
                    parts = date_str.split("-")
                    if parts and parts[0].isdigit():
                        pub_year = int(parts[0])
                except Exception:
                    pass
            
            # 6. Extract aspects and build dataframe records
            for lemma in lemmas:
                aspects = extract_aspects(
                    lemma,
                    text_comments=lemma.get("text_comments", [])
                )
                records.append({
                    "title": lemma["id"],
                    "problem":        aspects["aspect_statement"],
                    "method":         aspects["aspect_strategy"],
                    "finding":        aspects["aspect_dependencies"],
                    "interpretation": aspects["aspect_context"],
                    "theory": lemma["theory"],
                    "file": lemma["file"],
                    "line": lemma["line"],
                    "proof_text": lemma["proof_text"],
                    "statement_text": lemma["statement_text"],
                    "publication_year": pub_year,
                })
                
        except Exception as e:
            print(f"  Error processing theory {theory}: {e}")
            continue
            
    # 7. Post-process definition dependencies (lemmas that use this definition)
    compute_definition_dependencies(records)
    
    df = pd.DataFrame(records)
    print(f"Ingestion completed. Total lemmas ingested: {len(df)}")
    return df


def compute_definition_dependencies(records: list[dict[str, Any]]) -> None:
    """For each definition record, annotate its 'finding' with the names of lemmas that cite it.

    In Format B:
    - ``interpretation`` holds "keyword in Theory" (e.g. "definition in HOL.List").
    - ``finding`` holds cited dependency identifiers for lemmas, and is the column
      that gets populated here for definitions (listing their dependents).
    """
    definitions = []
    for r in records:
        interp = r.get("interpretation", "")
        # interpretation is now "<keyword> in <theory>"
        if interp.startswith(("definition in ", "fun in ", "function in ",
                              "primrec in ", "datatype in ", "type_synonym in ",
                              "inductive in ", "coinductive in ", "record in ",
                              "abbreviation in ")):
            title = r.get("title", "")
            name = title.split(".")[-1] if "." in title else title
            if name:
                definitions.append((r, name))

    for def_record, def_name in definitions:
        using_lemmas = []
        # Match bare name or name_def convention
        pattern = re.compile(rf'\b{re.escape(def_name)}(?:_def)?\b')

        for r in records:
            if r is def_record:
                continue
            # Only scan lemma records (non-definition constructs)
            if r.get("interpretation", "").startswith(("definition in ", "fun in ",
                                                        "function in ", "primrec in ",
                                                        "datatype in ", "type_synonym in ",
                                                        "inductive in ", "coinductive in ",
                                                        "record in ", "abbreviation in ")):
                continue

            stmt  = r.get("statement_text", "") or r.get("problem", "")
            proof = r.get("proof_text", "")
            deps  = r.get("finding", "")

            if pattern.search(stmt) or pattern.search(proof) or pattern.search(deps):
                using_lemmas.append(r.get("title", ""))

        if using_lemmas:
            def_record["finding"] = ", ".join(sorted(using_lemmas))
        else:
            def_record["finding"] = "none"

