# Reinforcement Learning for Conversational Memory Management

Adaptive Context Compression and Dynamic Memory Consolidation for LLM Agents

Learn a memory-retention policy for conversational agents under constrained context budgets using reinforcement learning.

This project frames memory management as a sequential decision-making problem:

> Which conversational messages are worth preserving under limited context capacity?

Instead of naive truncation or retrieval heuristics, the agent learns to:
- Retain important facts
- Discard irrelevant information
- Replace outdated memories (contradictions)
- Manage memory under pressure

---

# Core Idea

Conversational memory is modeled as an RL problem:

    incoming message
        ↓
    memory policy (RL agent)
        ↓
    KEEP / DROP / REPLACE_SIMILAR
        ↓
    future question answering reward

The agent learns selective compression of conversational history under a fixed memory budget.

---

# Current System (Implemented)

## Memory Actions

At each timestep, the policy selects:

- KEEP
- DROP
- REPLACE_SIMILAR

### KEEP
Store message in memory while respecting a fixed memory budget.

### DROP
Ignore message (no storage cost).

### REPLACE_SIMILAR
Overwrite the most semantically similar existing memory item.

This enables:
- Contradiction handling
- Stale memory replacement
- Semantic deduplication

Note: SUMMARIZE exists in earlier design iterations but is currently not part of the active training loop.

---

# Environment

A Gymnasium-based synthetic conversational environment.

Each episode contains:
- Factual statements
- Distractor messages
- Contradictions / updates
- Final query

Example:

    I live in Boston.
    I enjoy hiking.
    I moved to Seattle.
    ...
    Question: Where do I live now?

Correct answer:

    Seattle

---

# Curriculum Learning

Training uses an adaptive curriculum that gradually increases task difficulty.

| Level | Task type | Distractors |
|---|---|---|
| 0 | Simple factual recall | `base_distractors / 2` |
| 1 | Factual recall | `base_distractors` |
| 2 | Contradiction resolution | `base_distractors` |
| 3 | Distractor-heavy (fixed scenario) | high |
| 4 | Temporal updates (fixed scenario) | — |
| 5+ | Hard tasks (mixed) | — |

**Promotion / demotion** fires only after a full window of 20 episodes: promote if avg success > 75%, demote if < 30%. Evaluating on a partial window is suppressed to prevent single-sample cascades.

**Persistence**: curriculum state (level + success window) is saved to `curriculum_state.json` after every episode and restored on the next training run. Pass `curriculum_level=N` to `ConversationEnv` to pin to a fixed level without affecting saved state (used for benchmarking).

`base_distractors` is configurable per run, allowing the distractor density to be swept independently of curriculum level.

---

# Observation Space

The observation is a concatenated vector of four components:

| Component | Dim | Description |
|---|---|---|
| Current message embedding | 384 | `all-MiniLM-L6-v2` sentence embedding |
| Memory mean embedding | 384 | Mean-pooled embedding of all items currently in memory (zeros if empty) |
| Heuristic features | 7 | Number presence, location match, name match, message length, retrieval frequency, importance prior, temporal update marker |
| Metadata | 4 | Memory size, current step, avg memory age, max memory age |
| Per-slot similarities | `max_memory` | Cosine similarity between the current message and each memory slot (padded with zeros for empty slots) |

Default total: **784 dims** (with `max_memory=5`). Scales with `max_memory`.

Embeddings are produced using `sentence-transformers` / `all-MiniLM-L6-v2`, cached locally in `models/`.

---

# Memory Representation

Each memory item contains:

    {
        "message": str,
        "step": int,
        "fact_value": str | None,
        "id": int
    }

Memory is:
- Fixed-size (bounded by max memory)
- Updated via replacement policies
- Semantically searchable via embeddings

---

# Reward Function

Rewards are issued at two levels:

**Step reward** (every timestep):

    DROP a distractor (fact_value is None):  +0.1
    DROP a fact (fact_value is not None):    -0.2

**Terminal reward** (episode end):

    reward = (+10 if correct else -10) - len(memory) * 0.05

Correctness is determined by whether any retained memory item's `fact_value` matches the episode answer, including tuple-valued facts from multi-fact tasks.

The step rewards are small relative to the terminal signal (~2% of ±10) so they shape without dominating. They provide a dense training signal for distractor filtering, which is otherwise only rewarded implicitly through reduced memory cost.

---

# Task Types

## Retrieval Tasks

    My birthday is June 9.
    ...
    What is my birthday?

---

## Contradiction Tasks

    I live in Boston.
    I moved to Seattle.
    ...
    Where do I live now?

These tasks require:
- Memory overwrite
- Stale fact deletion
- Temporal reasoning

---

## Distractor-Heavy Tasks

Large amounts of irrelevant messages are inserted to test:
- Selective retention
- Compression robustness
- Noise filtering

---

## Temporal Update Tasks

    I live in Boston.
    I moved to Seattle.
    Now I live in Chicago.

These test:
- Temporal reasoning
- Recency handling
- Overwrite consistency

---

# Baselines

The system is evaluated against three rule-based policies in `memory/policies.py`:

| Baseline | Behavior |
|---|---|
| `KeepLastKPolicy` | Always KEEP until memory is full, then DROP |
| `RandomPolicy` | Uniform random action each step |
| `ReplaceSimilarPolicy` | REPLACE_SIMILAR if cosine similarity > 0.7, else KEEP |

---

# Evaluation Metrics

| Metric | Description |
|------|-------------|
| QA Accuracy | Correct final retrieval |
| Memory Tokens | Storage cost |
| Compression Ratio | Efficiency of retention |
| Contradiction Handling | Correct overwrite behavior |
| Retention Ratio | Fraction of kept messages |
| Stale Memory Rate | Failure to remove outdated facts |
| Reward | Final task objective |

---

# Architecture

## RL Agent

- PPO (Stable-Baselines3)
- MLP policy, configurable hidden layers (default `[256, 256]`)

Key `ConversationEnv` parameters:

| Parameter | Default | Description |
|---|---|---|
| `max_memory` | 5 | Memory budget (also sets obs dim) |
| `base_distractors` | 10 | Distractor count at curriculum level 1 |
| `curriculum_level` | `None` | Pin to fixed level (benchmarking); `None` resumes from checkpoint |
| `tb_log_dir` | `tensorboard/env` | TensorBoard output directory |

---

## Embeddings

- `sentence-transformers` / `all-MiniLM-L6-v2`
- Weights cached locally in `models/all-MiniLM-L6-v2/` (not committed); downloaded automatically on first run
- Single lazy-loaded global encoder instance per process

---

## Evaluation Stack

- pandas (aggregation)
- matplotlib (visualization)
- tensorBoard (training + memory stats)

---

# Project Structure

    rl-memory-agent/
    │
    ├── env/
    │   ├── conversation_env.py   # Gymnasium env; parameterized by max_memory, base_distractors
    │   ├── curriculum.py         # Adaptive difficulty; persistent checkpoint
    │   ├── tasks.py              # Synthetic conversation generators
    │   ├── features.py           # Embeddings, heuristics, obs vector builder
    │
    ├── memory/
    │   ├── policies.py           # Baseline rule-based policies
    │   ├── similarity.py         # Cosine similarity search over memory
    │   ├── memory_stats.py       # Per-episode diagnostic metrics
    │
    ├── training/
    │   ├── config.py             # TrainingConfig dataclass with per-run paths
    │   ├── train_ppo.py          # train(config) -> PPO; runnable as __main__
    │   ├── sweep.py              # Scaling experiment grid; python -m training.sweep
    │
    ├── evals/
    │   ├── benchmark.py          # benchmark_ppo / benchmark_all / benchmark_policy
    │   ├── metrics.py            # Episode-level metric functions
    │
    ├── plans/                    # Design docs and next-step plans
    ├── models/                   # Local embedder weights (gitignored)
    ├── results/                  # Per-run model checkpoints and eval summaries
    ├── tensorboard/              # TensorBoard logs (per run)
    └── requirements.txt

---

# Current Milestones

## Milestone 1
Rule-based memory system (keep-last-K baseline)

## Milestone 2
PPO-based memory policy (KEEP / DROP / REPLACE)

## Milestone 3
Embedding-based similarity memory replacement

## Milestone 4
Contradiction-aware memory updates

## Milestone 5
Curriculum learning + stability improvements

## Milestone 6 (current)
- Expanded observation space: memory mean embedding + heuristic features + per-slot similarities (388 → 784 dims)
- Dense intermediate rewards for step-level distractor filtering
- Fixed correctness signal for multi-fact tasks
- Curriculum persistence across training runs + promotion/demotion bug fix
- Parameterized environment and training config; scaling experiment sweep infrastructure

---

# Future Work

## LLM-Generated Environments

Replace synthetic task generation with LLM-generated conversations featuring:
- Natural language drift
- Implicit facts
- Realistic dialogue structure

---

## LLM-Based Reward Model

Replace rule-based evaluation with:
- Semantic correctness judging
- LLM-as-judge scoring
- Hallucination detection

---

## Advanced Memory Systems

- Memory graphs
- Hierarchical compression
- Learned decay functions
- Slot-based memory architectures

---

## Compression Analysis

Analyze:
- Accuracy vs memory budget
- Reward vs context length
- Retention vs noise ratio

to study emergent memory behavior.

---

# Research Direction

This project explores reinforcement learning as a mechanism for adaptive memory compression in language agents.

It connects:
- Reinforcement learning
- Natural language understanding
- Memory systems and compression
- Agentic LLM architectures

---

# Status

This is an active research prototype exploring:
- Learned memory policies
- Contradiction-aware compression
- Curriculum-driven RL training

It is not a production system, but a foundation for memory-augmented LLM agents with learned context management.