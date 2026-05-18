# Reinforcement Learning for Conversational Memory Management

Adaptive Context Compression and Dynamic Memory Consolidation for LLM Agents

Learn a memory-retention policy for conversational agents under constrained context budgets using reinforcement learning.

This project frames memory management as a sequential decision-making problem:

> Which conversational messages are worth preserving under limited context capacity?

Instead of naive truncation or retrieval heuristics, the agent learns to:
- retain important facts
- discard irrelevant information
- replace outdated memories (contradictions)
- manage memory under pressure

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
- contradiction handling
- stale memory replacement
- semantic deduplication

Note: SUMMARIZE exists in earlier design iterations but is currently not part of the active training loop.

---

# Environment

A Gymnasium-based synthetic conversational environment.

Each episode contains:
- factual statements
- distractor messages
- contradictions / updates
- final query

Example:

    I live in Boston.
    I enjoy hiking.
    I moved to Seattle.
    ...
    Question: Where do I live now?

Correct answer:

    Seattle

---

# Curriculum Learning (NEW)

Training uses an adaptive curriculum that gradually increases task difficulty.

Curriculum levels include:
- simple factual recall
- distractor-heavy sequences
- contradiction resolution
- temporal updates
- multi-hop retrieval tasks

Difficulty increases based on recent success rate, producing a staged learning progression:

1. memorization
2. filtering noise
3. contradiction resolution
4. long-context robustness

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
- sentence-transformers
- all-MiniLM-L6-v2

---

# Memory Representation

Each memory item contains:

    {
        "message": str,
        "step": int,
        "fact_value": Optional[str],
        "id": int
    }

Memory is:
- fixed-size (bounded by max memory)
- updated via replacement policies
- semantically searchable via embeddings

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
- correctness
- compression efficiency
- minimal unnecessary retention

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
- memory overwrite
- stale fact deletion
- temporal reasoning

---

## Distractor-Heavy Tasks

Large amounts of irrelevant messages are inserted to test:
- selective retention
- compression robustness
- noise filtering

---

## Temporal Update Tasks

    I live in Boston.
    I moved to Seattle.
    Now I live in Chicago.

These test:
- temporal reasoning
- recency handling
- overwrite consistency

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
- curriculum sampling
- episodic reward
- memory simulation
- contradiction-aware updates

---

## Embeddings

- sentence-transformers
- all-MiniLM-L6-v2

Used for:
- message representation
- similarity-based replacement

---

## Evaluation Stack

- Pandas (aggregation)
- Matplotlib (visualization)
- TensorBoard (training + memory stats)

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
- natural language drift
- implicit facts
- realistic dialogue structure

---

## LLM-Based Reward Model

Replace rule-based evaluation with:
- semantic correctness judging
- LLM-as-judge scoring
- hallucination detection

---

## Advanced Memory Systems

- memory graphs
- hierarchical compression
- learned decay functions
- slot-based memory architectures

---

## Scaling Experiments

Increase difficulty:

    distractors: 10 → 100 → 1000
    memory size: 5 → 20 → dynamic
    delay: short → long-horizon retrieval

Measure:
- accuracy vs compression curves
- robustness under noise
- contradiction resolution stability

---

## Compression Analysis

Analyze:
- accuracy vs memory budget
- reward vs context length
- retention vs noise ratio

to study emergent memory behavior.

---

# Research Direction

This project explores reinforcement learning as a mechanism for adaptive memory compression in language agents.

It connects:
- reinforcement learning
- natural language understanding
- memory systems and compression
- agentic LLM architectures

---

# Status

This is an active research prototype exploring:
- learned memory policies
- contradiction-aware compression
- curriculum-driven RL training

It is not a production system, but a foundation for memory-augmented LLM agents with learned context management.