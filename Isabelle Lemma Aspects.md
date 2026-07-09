# Isabelle Lemma Aspect Schema: Design Idea

## Overview

Isabelle lemmas are embedded into four semantic aspects that form an **epistemic trajectory** through concept space. The four aspects represent the genuinely distinct discourse domains of mathematical proof reasoning — not as a chronological narrative of proof composition, but as the four independent dimensions of mathematical knowledge encoded in a theorem.

Each consecutive pair of aspect embeddings defines a **displacement operator** (D). These operators are the primary analytic objects in EDEL theory, measuring the semantic distance traversed between phases of the reasoning process.

```
emb_P ──(D_pm)──► emb_M ──(D_mf)──► emb_F ──(D_fi)──► emb_I
  │                                                        │
  └────────────────── D_pi (epistemic closure) ────────────┘
```

---

## The Four Aspects

### 1. `problem` — Premises / Hypotheses

**Content**: The formal antecedents of the lemma — everything the prover must assume for the result to hold.

**Examples**:
```
lemma foo: "count A x = count B x ⟹ x ∈# A ⟹ A = B"
  → problem = "count A x = count B x, x ∈# A"

theorem bar:
  assumes "P" and "Q"
  shows "R"
  → problem = "P, Q"

lemma baz: "∀x. f x = g x"   (unconditional)
  → problem = "none"
```

**Extraction**: Parsed from `statement_text` without REPL interaction. For the `⟹`-chain form, all but the last top-level conjunct are premises. For the Isar `assumes`/`shows` form, the `assumes` clauses are collected. For unconditional lemmas, the field is empty (`"none"` — maps to a zero vector).

**Justification**: The premises define the *conditions* required to apply this lemma — the starting point of the theorem's epistemic content. Separating them from the conclusion is essential so that `D_pi` measures the full implication as a directed vector rather than a narrowing within a compound embedding.

---

### 2. `method` — Proof Skeleton

**Content**: The **declarative** layer of the proof — the intermediate claims, equational chains, case splits, and variable introductions made explicit by the prover.

**Isar keywords that produce method segments**:
```
proof   qed     have    show    also    finally
next    case    assume  fix     obtain  define   let
```

**Example** (from a structured Isar proof):
```
method segments:
  "proof-"
  "have \"(A2 ∪# ((A1 ∪# (X -# c1)) -# c2)) = (A2 ∪# (A1 ∪# ((X -# c1) -# c2)))\""
  "also have \"... = (A1 ∪# ((A2 ∪# (X -# c2)) -# c1))\""
  "finally show ?thesis"
  "qed"
```

**Extraction**: The Isabelle REPL's `Ir.source_map` assigns a keyword to every proof segment. Segments whose keyword is in the skeleton set above are routed to `method`. This requires no regex — the keyword is already available from `seg_map[idx]["keyword"]` in the parsed output.

**Degenerate case**: One-liner proofs (`by simp`, `apply auto`) produce no skeleton segments. `method = ""` for these lemmas — which is semantically correct: a flat proof has no intermediate structure to record.

**Justification**: The proof skeleton is the *declarative map* of the proof — what the prover claims step-by-step, independent of how those claims are verified. It is drawn from a fundamentally different vocabulary (Isabelle mathematical propositions) than the tactic layer, ensuring `D_mf` has genuine variance.

---

### 3. `finding` — Tactic Application

**Content**: The **operational** layer of the proof — the specific tactics, automation invocations, fact citations, and chaining operations used to actually close each goal.

**Isar keywords that produce finding segments**:
```
apply   by      using   unfolding  from    with
then    hence   thus    note       done    sorry   oops
```

**Example** (same structured proof as above):
```
finding segments:
  "using assms by auto"
  "using assms by auto"
  "by auto"
```

**Example** (apply-style proof):
```
finding segments:
  "apply (induction xs)"
  "apply (simp add: count_inject fun_eq_iff)"
```

**Extraction**: Same `seg_map` keyword filter as method — segments with operational keywords are routed to `finding`. No regex required.

**Degenerate case**: Definitions and axioms have no proof body, so `finding = ""`. This is correct: there is no operational execution to record for a definition.

**Justification**: The tactic layer captures *how* the proof was mechanically realised. It occupies a distinct semantic space from both the skeleton (mathematical propositions) and the premises/conclusion (the theorem's content). The `D_mf` operator measures how abstractly the automation relates to the claimed intermediate steps — e.g., high `D_mf` when the skeleton claims specific mathematical structure but all goals are closed by opaque `blast` or `metis`.

---

### 4. `interpretation` — Conclusion

**Content**: The formal consequent of the lemma — the mathematical result that holds if all premises are satisfied.

**Examples**:
```
lemma "count A x = count B x ⟹ A = B"
  → interpretation = "A = B"

theorem: assumes "P" shows "Q"
  → interpretation = "Q"

lemma "∀x. f x = g x"   (unconditional)
  → interpretation = "∀x. f x = g x"   (whole statement is the conclusion)
```

**Extraction**: Parsed from `statement_text` without REPL interaction. For the `⟹`-chain form, the last top-level conjunct is the conclusion. For `assumes`/`shows`, the `shows` clause is taken. For unconditional lemmas, the entire proposition is the conclusion.

**Justification**: The conclusion is the *result* the theorem establishes — what the agent gains by successfully applying it. It is always non-empty and is semantically distinct from the premises for any non-trivial theorem.

---

## The Epistemic Closure Operator: `D_pi`

The key insight of this design is that the full proposition `premises ⟹ conclusion` is encoded not as a **single embedding point** but as a **directed displacement vector**:

```
D_pi = emb_interpretation − emb_problem
     = emb_conclusion − emb_premises
```

This is the **implication vector** of the theorem — a geometric object representing the full semantic leap from conditions to achievement.

**Properties**:

- `‖D_pi‖` = **implication strength** / epistemic closure distance
  - Near zero → trivial consequence (conclusion is semantically close to premises)
  - Large → bridging theorem (conclusion is far from premises; premises and conclusion live in different mathematical domains)

- Two lemmas with the same premises but different conclusions: identical `emb_P`, different `D_pi` directions — correctly distinguishable.
- Two lemmas with the same conclusion but different premises: identical `emb_I`, different `D_pi` directions — also correctly distinguishable.
- A single full-statement embedding would conflate both cases.

For **unconditional lemmas** (no premises, `emb_P = 0`):
```
D_pi = emb_conclusion
‖D_pi‖ = ‖emb_conclusion‖  →  distinctiveness of the result in concept space
```
The result holds without any epistemic precondition. The magnitude measures how "peripheral" or "specific" the conclusion is relative to the centroid of the embedding space.

---

## Operator Semantics

| Operator | Formula | Interpretation |
|:---------|:--------|:---------------|
| `D_pm` | `‖emb_method − emb_problem‖` | How far the proof's intermediate structure departs from the stated conditions. High = proof introduces claims not directly reflected in the hypotheses. |
| `D_mf` | `‖emb_finding − emb_method‖` | How abstractly the automation relates to the declared structure. High = skeleton claims specific things; tactics close them by opaque automation. Low = tactics follow transparently from the skeleton. |
| `D_fi` | `‖emb_interp − emb_finding‖` | Proof harvest — how far the conclusion reaches beyond the tactic machinery. High = narrow tactics yield a broad result. |
| `D_pi` | `‖emb_interp − emb_problem‖` | **Implication strength / epistemic closure.** The primary characterisation of the theorem's significance. |

---

## Extraction Summary

| Aspect | Source | REPL Required? |
|:-------|:-------|:---------------|
| `problem` | Parse `statement_text` for `⟹` antecedents or `assumes` clauses | No |
| `method` | `Ir.source_map` segments with keywords: `have`, `show`, `also`, `case`, `fix`, `obtain`, `proof`, `qed`, etc. | Yes (source map) |
| `finding` | `Ir.source_map` segments with keywords: `apply`, `by`, `using`, `unfolding`, `from`, `then`, `hence`, etc. | Yes (source map) |
| `interpretation` | Parse `statement_text` for last `⟹` consequent or `shows` clause | No |

**Metadata (stored but not embedded)**:
- `statement_text` — full verbatim proposition for display in search results
- `cited_deps` — identifiers cited in `using`/`simp add:` for provenance tracing
- `construct_label` — `"lemma in Theory"` for filtering and display

