"""EDEL-RAG MCP server for Isabelle/AFP proof assistance."""

from __future__ import annotations

import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP, Context

from edel.isabelle.index import NumpyRAGIndex
from edel.io.llm import get_llm_client

# Initialize MCP Server
mcp = FastMCP("EDEL-RAG")

# Load Index
index = NumpyRAGIndex()
INDEX_DIR = os.getenv("EDEL_RAG_INDEX_DIR", "artifacts/rag_index")

try:
    index.load(INDEX_DIR)
except Exception as e:
    print(f"Warning: Could not load static RAG index from {INDEX_DIR}: {e}")
    print("EDEL-RAG will operate in session-only mode unless a static index is loaded.")


def get_embedding_client():
    """Build the embedding client from environment configuration."""
    provider = os.getenv("EDEL_EMBEDDING_PROVIDER", "openai")
    model = os.getenv("EDEL_EMBEDDING_MODEL", "text-embedding-3-large")
    api_key = os.getenv("VOYAGE_API_KEY" if provider == "voyage" else "OPENAI_API_KEY", "")
    
    config = {
        "provider": provider,
        "model": model,
        "api_key": api_key,
    }
    if provider == "voyage":
        config["input_type"] = "query"
        
    return get_llm_client(config)


def format_search_results(hits: list[dict]) -> str:
    """Format index hits into a readable Markdown block."""
    if not hits:
        return "No matching lemmas found."
        
    lines = []
    for i, hit in enumerate(hits):
        meta = hit["lemma"]
        lines.append(f"### {i+1}. `{meta['title']}` (Score: {hit['score']:.3f})")
        lines.append(f"- **Statement**: `{meta['problem']}`")
        if meta.get("finding") and meta["finding"] != "unknown":
            lines.append(f"- **Strategy**: `{meta['finding']}`")
        if meta.get("interpretation") and meta["interpretation"] != "none":
            lines.append(f"- **Dependencies**: `{meta['interpretation']}`")
        
        # Source location
        location = f"{meta['theory']}"
        if meta.get("file"):
            location += f" ({meta['file']}:{meta.get('line', '')})"
        lines.append(f"- **Location**: {location}")
        lines.append("")
        
    return "\n".join(lines)


@mcp.tool(description=(
    "Search for lemmas semantically similar to a goal or statement. "
    "Specify aspect='statement' to match by goal structure, 'strategy' to find similar proofs, "
    "'dependencies' to find lemmas using similar facts, 'context' for similar theories, "
    "or 'all' for a hybrid search across all aspects."
))
async def search_lemmas(
    query: str,
    aspect: str = "statement",  # "statement" | "context" | "strategy" | "dependencies" | "all"
    theory_filter: str = "",
    max_results: int = 10,
) -> str:
    """Perform semantic search on static and live session indices."""
    client = get_embedding_client()
    query_emb = client.generate_embedding(query)
    
    # Map tool aspects to index aspects
    aspect_map = {
        "statement": "problem",
        "context": "method",
        "strategy": "finding",
        "dependencies": "interpretation"
    }
    
    if aspect == "all":
        # Multi-aspect hybrid search
        all_results = {}
        for asp_name, idx_asp in aspect_map.items():
            hits = index.search(query_emb, aspect=idx_asp, max_results=max_results, theory_filter=theory_filter)
            for h in hits:
                lemma_id = h["lemma"]["title"]
                if lemma_id not in all_results or h["score"] > all_results[lemma_id]["score"]:
                    all_results[lemma_id] = h
        
        sorted_hits = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        hits = sorted_hits[:max_results]
    else:
        idx_asp = aspect_map.get(aspect, "problem")
        hits = index.search(query_emb, aspect=idx_asp, max_results=max_results, theory_filter=theory_filter)
        
    return format_search_results(hits)


@mcp.tool(description=(
    "Find proof strategies that worked for goals similar to the query goal. "
    "Returns ranked proof methods with example lemmas."
))
async def search_strategies(
    goal: str,
    max_results: int = 5,
) -> str:
    """Query statement index and aggregate strategies used by matching lemmas."""
    client = get_embedding_client()
    query_emb = client.generate_embedding(goal)
    
    hits = index.search(query_emb, aspect="problem", max_results=max_results * 3)
    
    strategies = {}
    for h in hits:
        strat = h["lemma"].get("finding", "unknown")
        if strat in ("unknown", "unknown (direct or simple)"):
            continue
        if strat not in strategies:
            strategies[strat] = []
        strategies[strat].append({
            "name": h["lemma"]["title"],
            "statement": h["lemma"]["problem"],
            "score": h["score"]
        })
        
    sorted_strats = sorted(strategies.items(), key=lambda x: max(e["score"] for e in x[1]), reverse=True)
    
    lines = ["### Suggested Proof Strategies", ""]
    for i, (strat, examples) in enumerate(sorted_strats[:max_results]):
        best_example = examples[0]
        lines.append(f"{i+1}. **Strategy**: `{strat}` (Confidence: {best_example['score']:.2f})")
        lines.append(f"   - *Example Lemma*: `{best_example['name']}`")
        lines.append(f"   - *Example Goal*: `{best_example['statement']}`")
        lines.append("")
        
    return "\n".join(lines)


@mcp.tool(description=(
    "Find lemmas semantically similar to a known lemma by its title (e.g. 'HOL.List.append_Nil')."
))
async def related_lemmas(
    lemma_name: str,
    max_results: int = 10,
) -> str:
    """Retrieve similar lemmas using the target lemma's pre-computed statement embedding."""
    target_idx = None
    for idx, meta in enumerate(index.metadata):
        if meta["title"].lower() == lemma_name.lower():
            target_idx = idx
            break
            
    if target_idx is None:
        return f"Lemma '{lemma_name}' not found in static RAG index."
        
    vector = index.embeddings["problem"][target_idx]
    hits = index.search(vector.tolist(), aspect="problem", max_results=max_results + 1)
    
    # Filter out target lemma itself
    hits = [h for h in hits if h["lemma"]["title"].lower() != lemma_name.lower()]
    return format_search_results(hits[:max_results])


@mcp.tool(description=(
    "Store a newly proven lemma in the session index. "
    "Call this after every successful proof to keep the agent's context fresh."
))
async def store_lemma(
    name: str,
    statement: str,
    proof_text: str,
    theory: str,
    dependencies: list[str] = [],
) -> str:
    """Parse and embed a new lemma, adding it to the runtime session index."""
    from edel.isabelle.aspects import extract_aspects
    
    lemma_dict = {
        "statement_text": f'lemma {name}: "{statement}"',
        "proof_text": proof_text,
        "theory": theory
    }
    
    # Extract aspects
    aspects = extract_aspects(lemma_dict)
    aspect_text_dict = {
        "problem": aspects["aspect_statement"],
        "method": aspects["aspect_context"],
        "finding": aspects["aspect_strategy"],
        "interpretation": aspects["aspect_dependencies"]
    }
    
    client = get_embedding_client()
    embeddings_dict = {}
    
    # Embed aspects
    for aspect_name, text in aspect_text_dict.items():
        if text.strip():
            embeddings_dict[aspect_name] = client.generate_embedding(text)
            
    # Set default zero embeddings for any empty aspects
    valid_emb = next((v for v in embeddings_dict.values() if v), None)
    dim = len(valid_emb) if valid_emb else 1536
    for k in ["problem", "method", "finding", "interpretation"]:
        if k not in embeddings_dict:
            embeddings_dict[k] = [0.0] * dim
            
    index.add_live_lemma(
        name=name,
        aspect_text_dict=aspect_text_dict,
        embeddings_dict=embeddings_dict,
        theory=theory,
        proof_text=proof_text,
        dependencies=dependencies
    )
    
    return f"Successfully stored lemma '{theory}.{name}' in RAG session index. It is now searchable."


@mcp.tool(description="List all lemmas added to the RAG session index during this session.")
async def session_lemmas() -> str:
    """Return all session lemmas."""
    if not index.live_metadata:
        return "No lemmas have been stored in this session yet."
        
    lines = ["### Session Lemmas", ""]
    for i, meta in enumerate(index.live_metadata):
        lines.append(f"{i+1}. `{meta['title']}`")
        lines.append(f"   - **Statement**: `{meta['problem']}`")
        lines.append(f"   - **Proof**: `{meta['proof_text']}`")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
