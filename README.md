# Reinforcement Learning for Conversational Memory Management

Adaptive Context Compression and Dynamic Memory Consolidation for LLM Agents

Learn a memory-retention policy for conversational agents under constrained context budgets.

This project explores reinforcement learning for:
- Conversational memory retention
- Semantic memory compression
- Contradiction resolution
- Token-efficient context management

The core question:

> Which memories are worth paying token cost for?

Instead of naive truncation or retrieval heuristics, the agent learns:
- What to remember
- What to forget
- What to summarize
- What to overwrite

It uses reward signals based on:
- Retrieval correctness
- Context efficiency
- Stale-memory penalties

---

# Motivation

LLM agents have limited context windows. Current approaches typically rely on:
- FIFO truncation
- Embedding retrieval
- Heuristics
- Summarization pipelines

This project frames conversational memory as a sequential decision-making problem:

```text
incoming message
    ↓
memory policy
    ↓
KEEP / DROP / SUMMARIZE / REPLACE
    ↓
future retrieval reward
```

The objective is not maximal retention but rather:
- Adaptive memory compression
- Semantic memory consolidation
- Efficient long-horizon recall

---

# Current Features

## Semantic Memory Policy

Observation space includes:
- Sentence embeddings
- Memory statistics
- Temporal metadata

using
- `sentence-transformers`
- `all-MiniLM-L6-v2`

The agent must infer memory importance semantically.

---

## Memory Actions

At each timestep the RL policy chooses:

- `KEEP`
- `DROP`
- `SUMMARIZE`
- `REPLACE_SIMILAR`

### KEEP
Store message in memory.

### DROP
Discard message entirely.

### SUMMARIZE
Compress message into lower-token representation.

Example:

```text
"My birthday is June 9 and I love cake."
↓
"Birthday: June 9"
```

### REPLACE_SIMILAR
Overwrite semantically related stale memories.

Example:

```text
"I live in Boston."
↓
"I moved to Seattle."
```

The system learns semantic memory updating rather than FIFO truncation.

---

# Environment

Synthetic multi-turn conversational environment built with Gymnasium.

Example:

```text
User: I live in Hoboken.
User: I enjoy hiking.
User: I moved to Seattle.
...
User: Where do I live now?
```

Correct answer:

```text
Seattle
```

---

# Reward Function

The policy optimizes:

- Retrieval accuracy
- Memory efficiency
- Contradiction resolution

Current reward:

```python
reward = (
    correct_answer * 10
    - (memory_tokens * 0.002)
    - incorrect_confident_answer * 5
)
```

Additional ideas:
- Hallucination penalties
- Latency penalties
- Summary compression bonuses
- Stale-memory penalties

---

# Tasks

## Retrieval Tasks

```text
"My birthday is June 9."
...
"What's my birthday?"
```

---

## Contradiction Tasks

```text
"I live in Boston."
...
"I moved to Seattle."
...
"Where do I live now?"
```

These tasks require:
- Memory replacement
- Stale memory deletion
- Temporal reasoning

---

## Long-Context Distraction

Large amounts of irrelevant conversation are inserted to test:
- Selective retention
- Compression
- Retrieval robustness

---

# Architecture

## Observation Space

Current observation includes:

```python
[
    current_message_embedding,
    memory_size,
    avg_memory_age,
    max_memory_age,
]
```

Future versions may include:
- Memory-summary embeddings
- Retrieval-frequency statistics
- Attention-based memory representations

---

## Memory Representation

Current memory entries:

```python
{
    "message": "...",
    "step": timestep,
    "summarized": False,
}
```

Future work:
- Memory slots
- Semantic memory graphs
- Hierarchical memory
- Learned memory decay

---

# Tech Stack

## RL

- `stable-baselines3`
- PPO

Future:
- `verl`
- distributed RLHF-style training
- GRPO / RLOO

---

## NLP

- `sentence-transformers`
- `all-MiniLM-L6-v2`

Future:
- TinyLlama
- SmolLM2
- Qwen2.5-0.5B-Instruct

---

## Environment

- Gymnasium custom environment

---

## Evaluation

- Pandas
- Matplotlib
- Weights & Biases

---

# Evaluation Metrics

Track:

| Metric | Description |
|---|---|
| QA Accuracy | Correct retrieval |
| Avg Context Tokens | Memory efficiency |
| Compression Ratio | Token reduction |
| Contradiction Accuracy | Correct stale-memory replacement |
| Retention Ratio | Fraction of stored messages |
| Latency | Inference cost |

---

# Baselines

Compare against:

- Keep last k
- FIFO truncation
- Random drop
- Embedding retrieval
- LRU memory
- Summarize oldest

---

# Current Progress

## Milestone 1
Rule-based memory manager.

## Milestone 2
PPO memory-retention policy.

## Milestone 3
Semantic embeddings replacing handcrafted labels.

## Milestone 4
Contradiction-aware memory updating.

## Milestone 5
Summarization-based compression.

---

# Project Structure

```text
rl-memory-agent/
│
├── env/
│   ├── conversation_env.py
│   ├── tasks.py
│   ├── features.py
│
├── memory/
│   ├── policies.py
│   ├── summarizer.py
│   ├── similarity.py
│
├── training/
│   ├── train_ppo.py
│
├── evals/
│   ├── benchmark.py
│   ├── metrics.py
│
├── notebooks/
├── results/
├── README.md
└── requirements.txt
```

---

# Future Directions

## Contextual Memory Policies

Current policy processes messages independently. Future policies may condition on:
- Compressed memory state
- Retrieval history
- Memory summaries

Example:

```python
[
    current_message_embedding,
    memory_summary_embedding,
]
```

This enables context-aware memory decisions and adaptive conversational state management.

---

## Memory Consolidation

Move beyond append-only memory toward:
- Semantic merging
- Memory rewriting
- Memory decay
- Hierarchical summaries

---

## Scaling Experiments

Increase distractor count:

```text
10 → 50 → 100 → 1000
```

Measure:
- Retrieval accuracy
- Compression efficiency
- Memory stability

---

## Compression Curves

Plot accuracy vs memory budget to analyze compression-retrieval tradeoffs, emergent memory policies, and semantic retention behavior.