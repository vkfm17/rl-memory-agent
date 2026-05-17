# Reinforcement Learning For Conversational Memory Management / Token-Efficient Conversational Memory via Reinforcement Learning

Learn memory retention policy for an LLM chatbot. This is a sequential compression problem where the agent learns, "Which memories are worth paying token cost for?"

Dynamic memory consolidation
Adaptive Context Compression for LLM Agents

## Components

### 1. Conversation Environment

A synthetic multi-turn conversation

User: My favorite color is blue.
User: I live in Hoboken.
...
User: What city do I live in?

Correct answer: Hoboken

### 2. Memory Manager (RL Agent)

At each step, decide
- KEEP message
- SUMMARIZE message
- DROP message

### 3. Chatbot Model

Tiny model/API (local Qwen2.5-0.5B-Instruct, SmolLM2, or TinyLlama)

### 4. Reward Function

- Balance correctness
- Token efficiency

```
python
reward = (
    correct_answer * 10
    - (memory_tokens * 0.002)
    - incorrect_confident_answer * 5
)
```

Other ideas:
- Hallucination penalties
- Latency penalities
- Summary compression bonuses

## Tech Stack

- stable-baselines3 to start, then verl for distributed RLHF-style training
- Rule-based QA oracle
- Tiny local HF model
- Gymnasium-style custom env

## Evaluation
- Pandas
- Matplotlib
- W&B
 
## Setup

Actions:
- KEEP
- DROP
- SUMMARIZE

Episode:
1. User message
2. Memory policy acts 
3. Chatbot responds
4. Environment checks correctness
5. Reward assigned

## Phase 1

Remember:
- Favorite movie
- Birthday
- Hometown

### Retrieval Task
"My birthday is June 9."
...
"What's my birthday?"

### Contradiction Task
"I moved from Boston to Seattle."
...
"Where do I live now?"

### Long-context distraction

Many irrelevant messages to learn selective retention

## MVP

### Milestone 1

Rule-based memory manager: keep latest 5 messages.

### Milestone 2

RL memory policy: train PPO/GRPO agent.

### Milestone 3

Add summarization action to compress older memories.

### Milestone 4

Evaluation:
- Native truncation
- Retrieval heuristics
- RL policy


## Structure

rl-memory-agent/
│
├── env/
│   ├── conversation_env.py
│   ├── tasks.py
│
├── memory/
│   ├── policies.py
│   ├── summarizer.py
│
├── training/
│   ├── train_ppo.py
│
├── evals/
│   ├── benchmark.py
│   ├── metrics.py
│
├── notebooks/
│
├── results/
│
├── README.md
└── requirements.txt


## Evaluation Metrics

Track:
- QA Accuracy: Did model remember correctly?
- Avg Context Tokens: Memory efficiency
- Compression Ratio: Saved tokens
- Contradiction Resolution: Updated stale memories
- Latency: Inference speed

Baselines:
- Keep last K
- FIFO truncation
- Random drop
- Embedding retrieval
- LRU memory
- Summarize oldest

## Future Improvements

Eventually move from per-message actions to memory graph/slots, where agent decides:
- Merge memories
- Rewrite memories
- Decay memories
