"""File-backed memory store for addressable fix buckets."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .models import (
    Bucket,
    BucketStats,
    CriticOutput,
    Episode,
    IndexEntry,
    MemoryIndex,
    MergeDecision,
)


class MemoryStore:
    """Manages the bucket library on disk.

    Layout:
        <root>/
            index.json          – compact index injected into Actor context
            addressables.jsonl   – append-only log of all addressable events
            buckets/
                <bucket_id>.json – full bucket content
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.buckets_dir = self.root / "buckets"
        self.index_path = self.root / "index.json"
        self.log_path = self.root / "addressables.jsonl"
        self._ensure_dirs()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        self.buckets_dir.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        """Wipe all memory (for experiment baselines)."""
        import shutil
        if self.root.exists():
            shutil.rmtree(self.root)
        self._ensure_dirs()
        self._write_index(MemoryIndex())

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def load_index(self) -> MemoryIndex:
        if not self.index_path.exists():
            idx = MemoryIndex()
            self._write_index(idx)
            return idx
        return MemoryIndex.model_validate_json(self.index_path.read_text())

    def load_index_text(self) -> str:
        """Return the index as paragraph summaries for prompt injection.

        Each entry is a short paragraph the Actor can scan to decide
        whether a bucket is worth opening via the ``read_bucket`` tool.
        """
        idx = self.load_index()
        if not idx.entries:
            return "(no addressable fixes stored yet)"
        blocks: list[str] = []
        for e in idx.entries:
            para = e.summary_paragraph or e.title
            blocks.append(f"[{e.bucket_id}] {e.title}\n{para}")
        return "\n\n".join(blocks)

    def _write_index(self, idx: MemoryIndex) -> None:
        self.index_path.write_text(idx.model_dump_json(indent=2))

    def _rebuild_index(self) -> MemoryIndex:
        entries: list[IndexEntry] = []
        for p in sorted(self.buckets_dir.glob("*.json")):
            b = Bucket.model_validate_json(p.read_text())
            entries.append(
                IndexEntry(
                    bucket_id=b.bucket_id,
                    title=b.title,
                    summary_paragraph=b.summary_paragraph,
                )
            )
        idx = MemoryIndex(entries=entries, version=self.load_index().version + 1)
        self._write_index(idx)
        return idx

    # ------------------------------------------------------------------
    # Bucket CRUD
    # ------------------------------------------------------------------

    def get_bucket(self, bucket_id: str) -> Bucket | None:
        p = self.buckets_dir / f"{bucket_id}.json"
        if not p.exists():
            return None
        return Bucket.model_validate_json(p.read_text())

    def _save_bucket(self, bucket: Bucket) -> None:
        p = self.buckets_dir / f"{bucket.bucket_id}.json"
        p.write_text(bucket.model_dump_json(indent=2))

    def list_bucket_ids(self) -> list[str]:
        return [p.stem for p in self.buckets_dir.glob("*.json")]

    # ------------------------------------------------------------------
    # Apply critic output (merge or create)
    # ------------------------------------------------------------------

    def apply_critic_output(
        self,
        critic: CriticOutput,
        episode: Episode,
        max_episodes: int = 10,
        max_drills: int = 5,
    ) -> Bucket:
        if critic.decision == MergeDecision.MERGE and critic.merge_target_bucket_id:
            bucket = self._merge_into(critic, episode, max_episodes, max_drills)
        else:
            bucket = self._create_new(critic, episode, max_drills)

        self._append_log(critic, episode)
        self._rebuild_index()
        return bucket

    def _create_new(
        self, critic: CriticOutput, episode: Episode, max_drills: int
    ) -> Bucket:
        bid = Bucket.make_id(critic.addressable_fix, critic.trigger_signals)
        existing = self.get_bucket(bid)
        if existing is not None:
            return self._merge_into_bucket(existing, critic, episode, max_drills)

        trace_summaries = [critic.trace_summary] if critic.trace_summary else []
        bucket = Bucket(
            bucket_id=bid,
            title=critic.addressable_fix[:80],
            addressable_fix=critic.addressable_fix,
            summary_paragraph=critic.summary_paragraph,
            trigger_signals=critic.trigger_signals,
            playbook=critic.playbook,
            trace_summaries=trace_summaries,
            when_to_open=critic.when_to_open,
            examples=[episode],
            synthetic_drills=critic.synthetic_drills[:max_drills],
            stats=BucketStats(hit_count=1),
        )
        self._save_bucket(bucket)
        return bucket

    def _merge_into(
        self,
        critic: CriticOutput,
        episode: Episode,
        max_episodes: int,
        max_drills: int,
    ) -> Bucket:
        target = self.get_bucket(critic.merge_target_bucket_id)  # type: ignore[arg-type]
        if target is None:
            return self._create_new(critic, episode, max_drills)
        bucket = self._merge_into_bucket(target, critic, episode, max_drills)
        self._save_bucket(bucket)
        return bucket

    @staticmethod
    def _merge_into_bucket(
        bucket: Bucket,
        critic: CriticOutput,
        episode: Episode,
        max_drills: int,
        max_trace_summaries: int = 6,
    ) -> Bucket:
        bucket.examples.append(episode)
        bucket.examples = bucket.examples[-max_drills:]

        new_triggers = set(bucket.trigger_signals) | set(critic.trigger_signals)
        bucket.trigger_signals = sorted(new_triggers)

        if critic.playbook:
            bucket.playbook = critic.playbook

        if critic.trace_summary and critic.trace_summary not in bucket.trace_summaries:
            bucket.trace_summaries.append(critic.trace_summary)
            bucket.trace_summaries = bucket.trace_summaries[-max_trace_summaries:]

        if critic.when_to_open:
            bucket.when_to_open = critic.when_to_open

        if critic.summary_paragraph:
            bucket.summary_paragraph = critic.summary_paragraph

        for drill in critic.synthetic_drills:
            if len(bucket.synthetic_drills) >= max_drills:
                break
            bucket.synthetic_drills.append(drill)

        bucket.stats.hit_count += 1
        bucket.stats.last_updated = time.time()
        if episode.outcome == "success":
            bucket.stats.success_count += 1

        return bucket

    # ------------------------------------------------------------------
    # Append-only log
    # ------------------------------------------------------------------

    def _append_log(self, critic: CriticOutput, episode: Episode) -> None:
        record = {
            "timestamp": time.time(),
            "decision": critic.decision.value,
            "addressable_fix": critic.addressable_fix,
            "task_id": episode.task_id,
            "outcome": episode.outcome,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # Bucket fetch for Actor selective-open
    # ------------------------------------------------------------------

    def fetch_bucket_context(
        self, bucket_id: str, max_drills: int = 3, max_episodes: int = 3
    ) -> str | None:
        """Return the full bucket content for the Actor's tool call.

        Includes playbook, drills, and full episodes (code + error traces).
        """
        bucket = self.get_bucket(bucket_id)
        if bucket is None:
            return None
        lines = [
            f"## Bucket: {bucket.title}",
            f"**Fix:** {bucket.addressable_fix}",
            "",
            "**Playbook:**",
        ]
        for i, step in enumerate(bucket.playbook, 1):
            lines.append(f"  {i}. {step}")

        if bucket.synthetic_drills:
            lines.append("")
            lines.append("**Drill examples:**")
            for d in bucket.synthetic_drills[:max_drills]:
                lines.append(f"  - {d.prompt}")
                lines.append(f"    Expected: {d.expected_behavior}")
                if d.test_code:
                    lines.append(f"    Test: {d.test_code}")

        if bucket.examples:
            lines.append("")
            lines.append(f"**Past episodes ({len(bucket.examples)} total, "
                         f"showing last {min(max_episodes, len(bucket.examples))}):**")
            for ep in bucket.examples[-max_episodes:]:
                lines.append("")
                lines.append(f"--- Episode: {ep.task_id} ({ep.outcome}) ---")
                lines.append(f"Diagnosis: {ep.diagnosis}")
                if ep.code:
                    lines.append(f"Code:\n```python\n{ep.code}\n```")
                if ep.error_trace:
                    lines.append(f"Error trace:\n{ep.error_trace}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Compaction (periodic dedup)
    # ------------------------------------------------------------------

    def compaction_candidates(self, threshold: float = 0.85) -> list[tuple[str, str]]:
        """Return pairs of bucket IDs whose trigger+fix text is suspiciously similar.

        Uses simple Jaccard on word sets as a v0 heuristic.
        """
        buckets = [self.get_bucket(bid) for bid in self.list_bucket_ids()]
        buckets = [b for b in buckets if b is not None]
        pairs: list[tuple[str, str]] = []
        for i, a in enumerate(buckets):
            for b in buckets[i + 1 :]:
                sim = _jaccard(
                    _words(a.addressable_fix, a.trigger_signals),
                    _words(b.addressable_fix, b.trigger_signals),
                )
                if sim >= threshold:
                    pairs.append((a.bucket_id, b.bucket_id))
        return pairs


def _words(fix: str, triggers: list[str]) -> set[str]:
    text = fix + " " + " ".join(triggers)
    return set(text.lower().split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
