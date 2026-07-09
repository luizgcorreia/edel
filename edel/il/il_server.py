"""I/L (Isabelle/Landscape) MCP server for Isabelle/AFP proof assistance."""

from __future__ import annotations

import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP, Context
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from edel.il.index import NumpyRAGIndex
from edel.io.llm import get_llm_client

# Initialize MCP Server
mcp = FastMCP("I/L")

# Load Index
index = NumpyRAGIndex()
INDEX_DIR = os.getenv("IL_INDEX_DIR", "artifacts/rag_index")

try:
    index.load(INDEX_DIR)
except Exception as e:
    print(f"Warning: Could not load static RAG index from {INDEX_DIR}: {e}")
    print("I/L will operate in session-only mode unless a static index is loaded.")


def get_embedding_client():
    """Build the embedding client from environment configuration."""
    provider = os.getenv("IL_EMBEDDING_PROVIDER", "voyage")
    model = os.getenv("IL_EMBEDDING_MODEL", "voyage-code-3")
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
        if meta.get("problem") and meta["problem"] != "none":
            lines.append(f"- **Premises**: `{meta['problem']}`")
        if meta.get("interpretation"):
            lines.append(f"- **Conclusion**: `{meta['interpretation']}`")
        if meta.get("method"):
            lines.append(f"- **Skeleton**:\n```isabelle\n{meta['method']}\n```")
        if meta.get("finding"):
            lines.append(f"- **Tactics**:\n```isabelle\n{meta['finding']}\n```")
        
        # Source location
        location = f"{meta.get('theory', '')}"
        if meta.get("file"):
            location += f" ({meta['file']}:{meta.get('line', '')})"
        lines.append(f"- **Location**: {location}")
        if meta.get("cited_deps") and meta["cited_deps"] != "none":
            lines.append(f"- **Cited Dependencies**: `{meta['cited_deps']}`")
        if meta.get("dependents_count") is not None:
            lines.append(f"- **Landscape Dependents Count**: `{meta['dependents_count']}`")
        lines.append("")
        
    return "\n".join(lines)


def format_definition_results(hits: list[dict]) -> str:
    """Format definition index hits into a readable Markdown block."""
    if not hits:
        return "No matching definitions found."
        
    lines = []
    for i, hit in enumerate(hits):
        meta = hit["definition"]
        lines.append(f"### {i+1}. `{meta['title']}` (Score: {hit['score']:.3f})")
        lines.append(f"- **Statement**: `{meta['problem']}`")
        
        location = f"{meta.get('theory', '')}"
        if meta.get("file"):
            location += f" ({meta['file']}:{meta.get('line', '')})"
        lines.append(f"- **Location**: {location}")
        if meta.get("dependents") and meta["dependents"] != "none":
            lines.append(f"- **Used in Lemmas**: `{meta['dependents']}`")
        if meta.get("dependents_count") is not None:
            lines.append(f"- **Landscape Dependents Count**: `{meta['dependents_count']}`")
        lines.append("")
        
    return "\n".join(lines)


@mcp.tool(description=(
    "Search for lemmas semantically similar to a query term or pattern. "
    "Set aspect='premises' to search by hypotheses/assumptions, "
    "'skeleton' to search by declarative proof structure/steps (have/show/case), "
    "'tactics' to search by operational tactics/commands (apply/by), "
    "'conclusion' to search by the final goal/lemma statement conclusion, "
    "or 'all' for a hybrid search across all aspects. "
    "Transitions can be performed by searching on one aspect and reading the others. "
    "For example, to find what tactics proved a goal: query the goal with aspect='conclusion', "
    "then read the 'Tactics' field in the returned results. "
    "Set sort_by_significance=True to bias search results toward widely cited, foundational lemmas. "
    "Set min_dependents=K to filter out obscure helper lemmas with fewer than K direct/transitive dependents."
))
async def search_lemmas(
    query: str,
    aspect: str = "conclusion",  # "premises" | "skeleton" | "tactics" | "conclusion" | "all"
    theory_filter: str = "",
    max_results: int = 10,
    sort_by_significance: bool = False,
    min_dependents: int = 0,
) -> str:
    """Perform semantic search on static and live session indices."""
    client = get_embedding_client()
    query_emb = client.generate_embedding(query)

    aspect_map = {
        "premises":     "problem",
        "skeleton":     "method",
        "tactics":      "finding",
        "conclusion":   "interpretation",
    }
    
    if aspect == "all":
        all_results = {}
        for asp_name, idx_asp in aspect_map.items():
            hits = index.search(
                query_emb,
                aspect=idx_asp,
                max_results=max_results,
                theory_filter=theory_filter,
                sort_by_significance=sort_by_significance,
                min_dependents=min_dependents
            )
            for h in hits:
                lemma_id = h["lemma"]["title"]
                if lemma_id not in all_results or h["score"] > all_results[lemma_id]["score"]:
                    all_results[lemma_id] = h
        
        sorted_hits = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        hits = sorted_hits[:max_results]
    else:
        idx_asp = aspect_map.get(aspect, "interpretation")
        hits = index.search(
            query_emb,
            aspect=idx_asp,
            max_results=max_results,
            theory_filter=theory_filter,
            sort_by_significance=sort_by_significance,
            min_dependents=min_dependents
        )
        
    return format_search_results(hits)


@mcp.tool(description=(
    "Search for definitions, types, or abbreviations in the dedicated Definition Space. "
    "Returns matching definitions by statement similarity. "
    "Set sort_by_significance=True to bias search results toward foundational/frequently cited entities. "
    "Set min_dependents=K to filter out obscure items used by fewer than K lemmas."
))
async def search_definitions(
    query: str,
    theory_filter: str = "",
    max_results: int = 10,
    sort_by_significance: bool = False,
    min_dependents: int = 0,
) -> str:
    """Perform semantic search on definitions in the Definition Space."""
    client = get_embedding_client()
    query_emb = client.generate_embedding(query)
    hits = index.search_definitions(
        query_emb,
        max_results=max_results,
        theory_filter=theory_filter,
        sort_by_significance=sort_by_significance,
        min_dependents=min_dependents
    )
    return format_definition_results(hits)


@mcp.tool(description=(
    "Find lemmas semantically similar to a known lemma by its title (e.g. 'HOL.List.append_Nil')."
))
async def related_lemmas(
    lemma_name: str,
    max_results: int = 10,
) -> str:
    """Retrieve similar lemmas using the target lemma's pre-computed conclusion embedding."""
    target_idx = None
    for idx, meta in enumerate(index.metadata):
        if meta["title"].lower() == lemma_name.lower():
            target_idx = idx
            break
            
    if target_idx is None:
        return f"Lemma '{lemma_name}' not found in static RAG index."
        
    # Query using conclusion embedding for related lemmas
    vector = index.embeddings["interpretation"][target_idx]
    hits = index.search(vector.tolist(), aspect="interpretation", max_results=max_results + 1)
    
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
    cited_deps: list[str] = [],
) -> str:
    """Parse and embed a new lemma, adding it to the runtime session index."""
    from edel.il.aspects import extract_aspects
    
    lemma_dict = {
        "statement_text": f'lemma {name}: "{statement}"',
        "proof_text": proof_text,
        "theory": theory,
        "keyword": "lemma"
    }
    
    aspects = extract_aspects(lemma_dict, text_comments=[])
    aspect_text_dict = {
        "problem":         aspects["aspect_statement"],
        "method":          aspects["aspect_strategy"],
        "finding":         aspects["aspect_dependencies"],
        "interpretation":  aspects["aspect_context"],
    }
    
    client = get_embedding_client()
    embeddings_dict = {}
    
    # Embed aspects
    for aspect_name, text in aspect_text_dict.items():
        if text.strip() and text != "none":
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
        cited_deps=cited_deps
    )
    
    return f"Successfully stored lemma '{theory}.{name}' in RAG session index. It is now searchable."


@mcp.tool(description=(
    "Store a newly defined construct (e.g. definition, fun, primrec, datatype) "
    "in the session definition index."
))
async def store_definition(
    name: str,
    statement: str,
    theory: str,
    dependents: list[str] = [],
) -> str:
    """Embed and store a new definition in the session Definition Space."""
    client = get_embedding_client()
    embedding = client.generate_embedding(statement)
    
    index.add_live_definition(
        name=name,
        statement_text=statement,
        embedding=embedding,
        theory=theory,
        dependents=", ".join(dependents) if dependents else "none"
    )
    return f"Successfully stored definition '{theory}.{name}' in the RAG session definition index."


@mcp.tool(description="Permanently persist all dynamically stored session lemmas and definitions to the on-disk static index.")
async def persist_session_lemmas() -> str:
    """Merge the in-memory session index into the on-disk index."""
    num_lemmas = len(index.live_metadata)
    num_defs = len(index.live_definition_metadata)
    if num_lemmas == 0 and num_defs == 0:
        return "No new session items to persist."
        
    try:
        index.persist_live_lemmas(INDEX_DIR)
        return f"Successfully persisted {num_lemmas} session lemmas and {num_defs} session definitions to the static index at '{INDEX_DIR}'."
    except Exception as e:
        return f"Failed to persist session items: {str(e)}"


@mcp.tool(description="List all lemmas and definitions added to the RAG session index during this session.")
async def session_lemmas() -> str:
    """Return all session lemmas and definitions."""
    lines = []
    if index.live_metadata:
        lines.append("### Session Lemmas")
        lines.append("")
        for i, meta in enumerate(index.live_metadata):
            lines.append(f"{i+1}. `{meta['title']}`")
            if meta.get("problem") and meta["problem"] != "none":
                lines.append(f"   - **Premises**: `{meta['problem']}`")
            lines.append(f"   - **Conclusion**: `{meta['interpretation']}`")
            lines.append("")
            
    if index.live_definition_metadata:
        lines.append("### Session Definitions")
        lines.append("")
        for i, meta in enumerate(index.live_definition_metadata):
            lines.append(f"{i+1}. `{meta['title']}`")
            lines.append(f"   - **Statement**: `{meta['problem']}`")
            lines.append("")
            
    if not lines:
        return "No lemmas or definitions have been stored in this session yet."
        
    return "\n".join(lines)


@mcp.prompt(name="il_proof_strategy", description="Instructions on using I/L (Isabelle/Landscape) during a proof session.")
def il_proof_strategy() -> str:
    """Provide structured guidelines for using I/L (Isabelle/Landscape)."""
    return (
        "You are an Isabelle/Isar assistant. You have access to the I/L (Isabelle/Landscape) vector index, "
        "which separates lemmas into 4 discourse spaces and definitions into a dedicated Definition Space:\n\n"
        "1. Premises (aspect='premises'): The assumptions, premises, or hypotheses of a lemma (defaults to 'none' if unconditional).\n"
        "2. Skeleton (aspect='skeleton'): The declarative Isar proof structure, containing skeleton steps (e.g., have/show/case/proof).\n"
        "3. Tactics (aspect='tactics'): The operational commands and tactics (e.g., apply/by/simp/auto/blast/metis).\n"
        "4. Conclusion (aspect='conclusion'): The final goal proposition proved by the lemma.\n\n"
        "To find proof breakthroughs, use 'discourse transitions' (cross-space querying):\n"
        "- **Find tactics for a target goal**: Query your target proposition using aspect='conclusion', and read the 'Tactics' and 'Skeleton' fields of the retrieved lemmas.\n"
        "- **Find lemmas with similar premises**: Query your assumptions/premises using aspect='premises'.\n"
        "- **Find useful or related definitions for a statement**: Query your statement (or parts of it) using the `search_definitions` tool. This will retrieve definition statements that are semantically close or relevant to your proposition (e.g., to discover definition names, types, or related constructs).\n"
        "- **Look up dependents**: Definitions include a list of 'dependents' (the names of lemmas in the archive that cite/use this definition), which can guide you to usage examples.\n"
    )


if __name__ == "__main__":
    mcp.run()
