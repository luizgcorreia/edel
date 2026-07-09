# Walkthrough: Landscape Height Integration & Epistemic Re-ranking

We have successfully implemented and verified the Phase 2 offline graph post-processing pipeline for the **Isabelle/Landscape (I/L)** RAG system. This adds support for calculating theorem and definition dependents transitively (landscape height) and using it inside the RAG server to bias and filter semantic search results.

---

## Changes Implemented

### 1. Landscape Height Post-Processing Script
* **File**: [compute_landscape_height.py](file:///home/correia/edel/edel/il/compute_landscape_height.py)
* **Functionality**:
  * Loads `metadata.parquet` and `definitions_metadata.parquet` from the target RAG index directory.
  * Builds the global Directed Acyclic Graph (DAG) of citations from lemma `cited_deps` and definition `dependents`.
  * Implements an $O(|V| + |E|)$ cycle-safe memoized DFS algorithm to calculate the transitive dependents count (landscape height) for each lemma and definition.
  * Updates both parquet metadata files with a new `dependents_count` column.

### 2. Integration into Indexing Pipelines
* **Files**: [build_afp_index.py](file:///home/correia/edel/scripts/build_afp_index.py) and [build_il_index.py](file:///home/correia/edel/edel/il/build_il_index.py)
* **Functionality**: Automatically triggers the landscape height calculation as a final post-processing pass at the end of the index building process.

### 3. Epistemic Prior search re-ranking
* **File**: [index.py](file:///home/correia/edel/edel/il/index.py) (`NumpyRAGIndex`)
* **Functionality**:
  * Loads the `dependents_count` field during startup (defaults to `0` for backward compatibility).
  * Exposes `min_dependents` parameter to filter out obscure items with low dependents count.
  * Exposes `sort_by_significance` parameter to apply log-prior re-ranking:
    $$Score(L) = S_{\text{semantic}}(L) + 0.15 \cdot \frac{\log(1 + H(L))}{\max \log(1 + H)}$$

### 4. MCP Server tool parameters
* **File**: [il_server.py](file:///home/correia/edel/edel/il/il_server.py)
* **Functionality**: Exposed `sort_by_significance` and `min_dependents` to both `search_lemmas` and `search_definitions` MCP tools so that the agent can actively configure RAG ranking priorities.

---

## Verification Results

### Automated Tests
* **File**: [test_il_landscape_height.py](file:///home/correia/edel/tests/test_il_landscape_height.py)
* Verified:
  * Proper transitive reachability calculation for a mock dependency DAG:
    * `Test.lemma_C` (has dependents B and A) $\to$ `dependents_count = 2`.
    * `Test.lemma_B` (has dependent A) $\to$ `dependents_count = 1`.
    * `Test.lemma_A` (has no dependents) $\to$ `dependents_count = 0`.
    * `Test.def_D` (has dependent A) $\to$ `dependents_count = 1`.
  * `min_dependents` filtering logic in `search()`.
  * Cosine similarity boost re-ranking with `sort_by_significance=True`.

Run result:
```bash
/home/correia/miniforge3/envs/edel/bin/python -m pytest tests/test_il_landscape_height.py -v
============================== 1 passed in 0.68s ===============================
```

### Full I/L Test Suite
All 33 `il` tests pass successfully:
```bash
/home/correia/miniforge3/envs/edel/bin/python -m pytest tests/ -k il -v
====================== 33 passed, 93 deselected in 15.36s ======================
```
