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

Curriculum levels include:
- Simple factual recall
- Distractor-heavy sequences
- Contradiction resolution
- Temporal updates
- Multi-hop retrieval tasks

Difficulty increases based on recent success rate, producing a staged learning progression:

1. Memorization
2. Filtering noise
3. Contradiction resolution
4. Long-context robustness

---

# Observation Space

The agent observes:

    [
        current_message_embedding,
        memory_size,
        avg_memory_age,
        max_memory_age
    ]

Embeddings are produced using:
- `sentence-transformers`
- `all-MiniLM-L6-v2`

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

The reward is computed at episode end:

    reward = (
        +10 if final answer is correct else -10
        - memory_token_cost
    )

Where:

    memory_token_cost ∝ total stored messages

This encourages:
- Correctness
- Compression efficiency
- Minimal unnecessary retention

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

The system is evaluated against:

- Keep last K messages
- FIFO truncation
- Random drop
- Embedding similarity retention
- LRU-style memory
- Heuristic contradiction replacement

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
- MLP policy with [256, 256] hidden layers

---

## Environment

Gymnasium custom environment with:
- Curriculum sampling
- Episodic reward
- Memory simulation
- Contradiction-aware updates

---

## Embeddings

- `sentence-transformers`
- `all-MiniLM-L6-v2`

Used for message representation and similarity-based replacement.

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
    │   ├── conversation_env.py
    │   ├── curriculum.py
    │   ├── tasks.py
    │   ├── features.py
    │
    ├── memory/
    │   ├── policies.py
    │   ├── similarity.py
    │   ├── memory_stats.py
    │
    ├── training/
    │   ├── train_ppo.py
    │
    ├── evals/
    │   ├── benchmark.py
    │   ├── metrics.py
    │
    ├── results/
    ├── notebooks/
    ├── tensorboard/
    ├── README.md
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

## Milestone 5 (current)
Curriculum learning + stability improvements

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

## Scaling Experiments

Increase difficulty:

    distractors: 10 → 100 → 1000
    memory size: 5 → 20 → dynamic
    delay: short → long-horizon retrieval

Measure:
- Accuracy vs compression curves
- Robustness under noise
- Contradiction resolution stability

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