# RSI — Runtime Self-Improvement

Test-time self-improvement loop via critique-maintained memory and synthetic drills. No retries, no weight updates — improvement happens across tasks through a growing library of reusable fix patterns.

## Architecture

```
Task ──► Actor (LLM) ──► Evaluator (sandbox) ──► Critic (LLM) ──► Memory Store
            ▲                                          │
            └──────────── index.json ◄─────────────────┘
                         (paragraphs)
```

**Actor** generates code conditioned on paragraph summaries of known fix patterns. Can call `read_bucket(id)` to fetch full details (playbook, drills, past code + traces) when a pattern looks relevant.

**Evaluator** runs code in a sandboxed subprocess with timeout. Returns structured pass/fail + full error traces.

**Critic** analyzes each attempt and produces a reusable addressable-fix abstraction: a summary paragraph, playbook, trigger signals, and synthetic drills. Decides whether to merge into an existing bucket or create a new one.

**Memory Store** is file-backed (`memory/buckets/*.json` + `memory/index.json`). Two-layer design: index stays small (paragraph summaries); full bucket content (code, traces, playbook) is loaded on demand.

## How It Works

1. Each task gets **one attempt** (no retries).
2. Actor sees the task + index of paragraph summaries.
3. If a bucket looks relevant, Actor calls `read_bucket(id)` to get full content (playbook, past code, error traces).
4. Actor generates code. Evaluator runs it.
5. Critic analyzes the result (pass or fail), produces a fix abstraction, updates memory.
6. Next task sees the updated index. Patterns carry over across tasks.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

## Usage

```bash
# Full experiment: baseline (no memory) vs memory-enabled
python run.py --benchmark humaneval --limit 20

# Use a different model
python run.py --actor-model gpt-4o-mini --critic-model gpt-4o --limit 10

# Point to a local/custom API
python run.py --base-url http://localhost:8000/v1 --actor-model my-model

# Verbose logging
python run.py --limit 5 -v
```

## Memory Layout

```
memory/
├── index.json            # Paragraph summaries (what the Actor always sees)
├── addressables.jsonl    # Append-only event log
└── buckets/
    └── <bucket_id>.json  # Full bucket: playbook, drills, episodes with code + traces
```

## Key Concepts

- **Two-layer context**: Index (paragraph summaries, always in context) + Buckets (full content, loaded on demand via tool call). Context stays small.
- **Addressable fix**: A reusable pattern (summary paragraph + playbook + trigger signals) extracted from a failure or success episode.
- **Bucket**: A semantic container grouping equivalent fixes; merged by *fix equivalence*, not string matching. Stores full code and error traces from past episodes.
- **Synthetic drills**: Generated mini-problems that exercise the same playbook with varied surface features.
- **Cross-task learning**: No retries. Patterns from Task 1 help on Task 50 as the index grows.
