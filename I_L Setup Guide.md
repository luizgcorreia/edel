# I/L (Isabelle/Landscape) RAG: Complete End-to-End Setup Guide

This guide covers setting up the entire I/L (Isabelle/Landscape) RAG assistant environment from scratch: cloning repositories, installing Isabelle2025-2, configuring the Python environment, building the recorded heap, running the pipeline to ingest/embed lemmas, and integrating the final system with Claude Code or another MCP-compatible agent.

---

## Prerequisites
* **Operating System**: Linux (tested on Ubuntu/Debian).
* **Conda/Mamba**: Conda environment manager (Miniforge3/Mambaforge is recommended).
* **API Keys**: A Voyage AI API key (`VOYAGE_API_KEY`) or OpenAI API key (`OPENAI_API_KEY`).

---

## Step 1: Clone Repositories and AFP
1. Clone the `edel` repository:
   ```bash
   git clone https://github.com/luizgcorreia/edel.git
   cd edel
   ```
2. Clone `AutoCorrode` inside the repository:
   ```bash
   git clone https://github.com/luizgcorreia/AutoCorrode.git
   ```
3. **Download/Clone the Archive of Formal Proofs (AFP)**:
   You have two options to obtain the AFP:
   * **Option A (Automatic via EDEL Provider)**: If you have Mercurial (`hg`) installed, EDEL's built-in `AFPProvider` can automatically clone the repository from Heptapod to `external/afp-2025-2` when running the pipeline.
   * **Option B (Manual Clone)**: Clone it manually using Mercurial:
     ```bash
     mkdir -p external
     hg clone https://foss.heptapod.net/isa-afp/afp-2025-2 external/afp-2025-2
     ```

---

## Step 2: Install Isabelle2025-2
I/L requires Isabelle2025-2 because it relies on the `record_theories` proof recording feature:
1. Download the Isabelle2025-2 Linux bundle:
   ```bash
   wget https://isabelle.in.tum.de/website-Isabelle2025-2/dist/Isabelle2025-2_linux.tar.gz
   tar -xzf Isabelle2025-2_linux.tar.gz
   ```
2. Add Isabelle to your environment `PATH` (e.g., in your `~/.bashrc`):
   ```bash
   export PATH="/path/to/Isabelle2025-2/bin:$PATH"
   ```
3. Register the AFP component with Isabelle:
   ```bash
   isabelle components -u /path/to/edel/external/afp-2025-2
   ```

---

## Step 3: Configure Conda & Install Dependencies
1. Create and activate the `edel` Conda environment:
   ```bash
   conda create -n edel python=3.11 -y
   conda activate edel
   ```
2. Install dependencies for the `AutoCorrode` I/R daemon:
   ```bash
   pip install -r AutoCorrode/ir/requirements.txt
   ```
3. Install the `edel` package in editable mode:
   ```bash
   pip install -e .
   ```

---

## Step 4: Build Isabelle Heap with Recorded Proof States
For the ingestion pipeline to parse proof steps and identify local strategies, proof states must be recorded in the Isabelle session heap:
```bash
isabelle build -b -o record_theories=true -d /path/to/edel/external/afp-2025-2/thys -j 8 HOL-Library
```
*Note: You can replace `HOL-Library` with a specific AFP entry or session you want to compile. The `-o record_theories=true` flag is crucial, as it tells the prover to record command spans in the session database.*

---

## Step 5: Process Ingestion and Build the Static RAG Index

Depending on your goals, you can build the index for a specific session/theory (e.g. for development or specific testing), or run the automated script to index the entire Archive of Formal Proofs (AFP).

### Method A: Build Index for a Specific Session/Theory (Quick/Development)
1. **Start the I/R REPL Daemon**: Spin up the Isabelle REPL daemon in a terminal so that the parser can programmatically query the theory segments:
   ```bash
   python AutoCorrode/ir/repl.py \
     --isabelle /path/to/Isabelle2025-2/bin/isabelle \
     --session HOL-Library \
     --mcp
   ```
   *Take note of the TCP authentication token printed on startup (e.g. `IR_Repl.token: abc123xyz`).*

2. **Run the Ingestion Pipeline**: In a separate terminal, export the environment credentials and run the index builder:
   ```bash
   export IR_AUTH_TOKEN="abc123xyz" # The token printed by repl.py
   export VOYAGE_API_KEY="your-voyage-key"
   
   # Run the build script (filters for a specific theory for illustration)
   python -m edel.il.build_il_index \
     --provider voyage \
     --model voyage-code-3 \
     --filter "Multiset" \
     --output artifacts/rag_index
   ```
   This generates `metadata.parquet`, `embeddings.npz`, `definitions_metadata.parquet`, and `definitions_embeddings.npz` inside `artifacts/rag_index/`.

### Method B: Build Index for the Entire AFP (Incremental/Automated)
For a production deployment, you can use the automated script `scripts/build_afp_index.py` which compiles the required session heaps on-demand, launches the REPL, and incrementally indexes and embeds the entire AFP:
1. Export the environment credentials:
   ```bash
   export VOYAGE_API_KEY="your-voyage-key"
   ```
2. Run the automated script:
   ```bash
   python scripts/build_afp_index.py \
     --isabelle /path/to/Isabelle2025-2/bin/isabelle \
     --afp-dir /path/to/edel/external/afp-2025-2/thys \
     --provider voyage \
     --model voyage-code-3 \
     --output artifacts/rag_index
   ```
   *Note: This script automatically tracks progress in `artifacts/rag_index/progress.json` and supports resuming from where it was interrupted. It will verify and build each session heap as needed.*


---

## Step 6: Start the RAG MCP Server
Verify and launch the server using the helper script:
1. Configure environment variables in `start_il_server.sh` or export them:
   ```bash
   export VOYAGE_API_KEY="your-voyage-key"
   export IL_EMBEDDING_PROVIDER="voyage"
   export IL_EMBEDDING_MODEL="voyage-code-3"
   export IL_INDEX_DIR="/path/to/edel/artifacts/rag_index"
   ```
2. Make the script executable and launch the server:
   ```bash
   chmod +x start_il_server.sh
   ./start_il_server.sh
   ```
   The server will print that it has loaded the static index and is waiting for connections on stdin.

---

## Step 7: Configure MCP Client (e.g., Claude Code)
To connect your agent to both the active prover (REPL) and the semantic knowledge graph (RAG), add them as servers in the agent's config (e.g., `~/.config/claude/mcp_config.json`):

```json
{
  "mcpServers": {
    "isabelle-repl": {
      "command": "/home/correia/miniforge3/envs/edel/bin/python",
      "args": [
        "/home/correia/edel/AutoCorrode/ir/repl.py",
        "--isabelle", "/home/correia/Isabelle2025-2/bin/isabelle",
        "--session", "HOL-Library",
        "--mcp"
      ]
    },
    "isabelle-landscape": {
      "command": "/home/correia/miniforge3/envs/edel/bin/python",
      "args": ["-m", "edel.il.il_server"],
      "env": {
        "VOYAGE_API_KEY": "your-voyage-key",
        "IL_EMBEDDING_PROVIDER": "voyage",
        "IL_EMBEDDING_MODEL": "voyage-code-3",
        "IL_INDEX_DIR": "/home/correia/edel/artifacts/rag_index",
        "PYTHONPATH": "/home/correia/edel"
      }
    }
  }
}
```

Now, when you spawn your agent (e.g., via `claude`), it has parallel access to the live REPL tools (to step through proofs) and the RAG tools (to retrieve lemmas by aspects, search definitions, and save newly proven helper lemmas and definitions).
