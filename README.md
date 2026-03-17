# RSI — Recusrive Self-Improvement

An LLM solves coding tasks one at a time. After each attempt (pass or fail), a second LLM extracts a reusable fix pattern and stores it in memory. Future tasks see those patterns and can pull up the full details if they look relevant. No retries on the same task, no fine-tuning.

## How it works

```
Task ──► Actor ──► Evaluator ──► Critic ──► Memory
            ▲                                  │
            └──────── index.json ◄─────────────┘
```

- **Actor**: solves the task. Sees a short index of known fix patterns. Can call `read_bucket(id)` to get full details (playbook, actual past code, error traces) if something looks relevant.
- **Evaluator**: runs code in a subprocess, returns pass/fail + tracebacks.
- **Critic**: looks at the attempt, extracts a generalizable fix pattern, decides whether to merge it into an existing bucket or create a new one.
- **Memory**: file-backed. `index.json` has paragraph summaries (small, always in context). `buckets/*.json` has the full content (loaded only when the Actor asks).

The key idea: the index is cheap to keep in context. The heavy stuff (playbooks, full code, full traces) lives in buckets and only gets loaded via tool call when the Actor thinks it's useful.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

## Usage

```bash
# baseline (no memory) vs memory-enabled, 20 HumanEval tasks
python run.py --benchmark humaneval --limit 20

# different models for actor and critic
python run.py --actor-model gpt-4o-mini --critic-model gpt-4o --limit 10

# local API
python run.py --base-url http://localhost:8000/v1 --actor-model my-model

# verbose
python run.py --limit 5 -v
```

## Memory layout

```
memory/
├── index.json            # paragraph summaries (what the Actor always sees)
├── addressables.jsonl    # append-only event log
└── buckets/
    └── <id>.json         # full bucket: playbook, drills, episodes w/ code + traces
```

## Concepts

- **Bucket**: groups equivalent fixes. Merged by whether the same playbook would work, not by string matching. Stores full code and error traces.
- **Index**: paragraph summaries of all buckets. Always in context. The Actor reads these to decide if anything is worth opening.
- **Addressable fix**: the reusable pattern extracted from an episode — a summary, playbook steps, trigger signals.
- **Synthetic drills**: small generated problems that target the same fix with different surface features.
- **Cross-task**: one attempt per task, no retries. Patterns from early tasks help on later ones.
