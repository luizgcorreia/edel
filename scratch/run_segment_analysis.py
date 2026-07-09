import subprocess
import sys
import re
import os
import time
import socket
from pathlib import Path
import numpy as np

# Adjust path to find edel package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edel.il.parser import parse_source_segments, group_segments_to_lemmas
from edel.il.ingest import EphemeralReplClient

def count_tokens(text):
    # Splits by word characters and non-word non-space characters
    return len(re.findall(r'\w+|[^\w\s]', text))

def run_analysis():
    isabelle_path = "/home/correia/Isabelle2025-2/bin/isabelle"
    afp_thys_dir = "/home/correia/edel/external/afp-2025-2/thys"
    session = "HOL-Library"
    port = 9155
    
    print(f"Starting REPL daemon for session '{session}' on port {port}...")
    repl_proc = subprocess.Popen([
        sys.executable, "AutoCorrode/ir/repl.py",
        "--isabelle", isabelle_path,
        "--session", session,
        "--dir", afp_thys_dir,
        "--port", str(port),
        "--mcp"
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    token = None
    start_time = time.time()
    
    # Read stdout line-by-line to get the auth token
    try:
        while time.time() - start_time < 90:
            line = repl_proc.stdout.readline().decode("utf-8", errors="replace")
            if not line:
                break
            print(f"  [REPL] {line.strip()}")
            m = re.search(r'Tcp_Handler: listening on 127\.0\.0\.1:(\d+)\s+\(token "([^"]+)"\)', line)
            if m:
                token = m.group(2)
                break
            m2 = re.search(r'IR_Repl\.token:\s+(\S+)', line)
            if m2:
                token = m2.group(1)
                break
    except Exception as e:
        print(f"Error reading REPL startup: {e}")
        
    if not token:
        print("[ERROR] Failed to start REPL or retrieve token. Exiting.")
        repl_proc.terminate()
        repl_proc.wait()
        return
        
    print(f"Connected to REPL. Token: {token}")
    client = EphemeralReplClient(host="127.0.0.1", port=port, token=token)
    
    try:
        # Disable truncation
        client.send("Ir.config (fn cfg => {color = #color cfg, show_ignored = #show_ignored cfg, full_spans = true, show_theory_in_source = #show_theory_in_source cfg, auto_replay = #auto_replay cfg});")
        
        # Get theories
        raw_thys = client.send("Ir.theories ();")
        theories = [t.strip() for t in raw_thys.splitlines() if t.strip()]
        # Filter for Multiset theory
        huffman_thys = [t for t in theories if t == "HOL-Library.Multiset"]
        print(f"Theories in HOL session: {huffman_thys}")
        
        for theory in huffman_thys:
            print(f"\n--- Analyzing Theory: {theory} ---")
            raw_source = client.send(f'Ir.source "{theory}" 0 ~1;')
            raw_map = client.send(f'Ir.source_map "{theory}" 0 ~1;')
            
            segments = parse_source_segments(raw_source)
            
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
            
            lemmas = group_segments_to_lemmas(seg_map, segments)
            
            # Segment level statistics
            seg_texts = list(segments.values())
            seg_char_lens = [len(s) for s in seg_texts]
            seg_token_lens = [count_tokens(s) for s in seg_texts]
            
            print(f"Raw Segment Count: {len(seg_texts)}")
            print(f"Raw Segment Char Lengths: Mean={np.mean(seg_char_lens):.1f}, Median={np.median(seg_char_lens)}, Min={min(seg_char_lens)}, Max={max(seg_char_lens)}")
            print(f"Raw Segment Token Lengths: Mean={np.mean(seg_token_lens):.1f}, Median={np.median(seg_token_lens)}, Min={min(seg_token_lens)}, Max={max(seg_token_lens)}")
            
            # Grouped Lemma level statistics
            print(f"\nGrouped Lemma/Definition Count: {len(lemmas)}")
            
            stmt_char_lens = [len(l["statement_text"]) for l in lemmas]
            stmt_token_lens = [count_tokens(l["statement_text"]) for l in lemmas]
            
            proof_char_lens = [len(l["proof_text"]) for l in lemmas if l["proof_text"]]
            proof_token_lens = [count_tokens(l["proof_text"]) for l in lemmas if l["proof_text"]]
            
            segments_per_lemma = []
            for l in lemmas:
                # count segments between start and end
                indices = sorted(segments.keys())
                start_i = indices.index(l["segment_start"])
                end_i = indices.index(l["segment_end"])
                segments_per_lemma.append(end_i - start_i + 1)
                
            print(f"Segments per Lemma: Mean={np.mean(segments_per_lemma):.1f}, Max={max(segments_per_lemma)}")
            print(f"Statement Char Lengths: Mean={np.mean(stmt_char_lens):.1f}, Median={np.median(stmt_char_lens)}, Min={min(stmt_char_lens)}, Max={max(stmt_char_lens)}")
            print(f"Statement Token Lengths: Mean={np.mean(stmt_token_lens):.1f}, Median={np.median(stmt_token_lens)}, Min={min(stmt_token_lens)}, Max={max(stmt_token_lens)}")
            
            if proof_char_lens:
                print(f"Proof Char Lengths: Mean={np.mean(proof_char_lens):.1f}, Median={np.median(proof_char_lens)}, Min={min(proof_char_lens)}, Max={max(proof_char_lens)}")
                print(f"Proof Token Lengths: Mean={np.mean(proof_token_lens):.1f}, Median={np.median(proof_token_lens)}, Min={min(proof_token_lens)}, Max={max(proof_token_lens)}")
            else:
                print("No proofs found (only definitions).")
                
            # Let's inspect some of the longest statements and proofs to check complexity
            print("\nTop 3 Longest Statement Texts:")
            longest_stmts = sorted(lemmas, key=lambda l: len(l["statement_text"]), reverse=True)[:3]
            for idx, l in enumerate(longest_stmts, 1):
                print(f"  {idx}. ID: {l['id']} ({len(l['statement_text'])} chars, {count_tokens(l['statement_text'])} tokens)")
                print(f"     Text: {l['statement_text']}")
                
            print("\nTop 3 Longest Proof Texts:")
            longest_proofs = sorted(lemmas, key=lambda l: len(l["proof_text"]), reverse=True)[:3]
            for idx, l in enumerate(longest_proofs, 1):
                print(f"  {idx}. ID: {l['id']} ({len(l['proof_text'])} chars, {count_tokens(l['proof_text'])} tokens)")
                print(f"     Text: {l['proof_text']}")
                
    finally:
        print("\nStopping REPL daemon...")
        repl_proc.terminate()
        try:
            repl_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            repl_proc.kill()
            repl_proc.wait()

if __name__ == "__main__":
    run_analysis()
