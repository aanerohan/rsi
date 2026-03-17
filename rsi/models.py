"""Pydantic data models for the self-improvement loop."""

from __future__ import annotations

import hashlib
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class TestStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    TIMEOUT = "timeout"
    ERROR = "error"


class FailedTest(BaseModel):
    input: str = ""
    expected: str = ""
    actual: str = ""
    assertion: str = ""
    traceback: str = ""


class EvalResult(BaseModel):
    task_id: str
    status: TestStatus
    failed_tests: List[FailedTest] = Field(default_factory=list)
    runtime_ms: float = 0.0
    stdout: str = ""
    stderr: str = ""


# ---------------------------------------------------------------------------
# Bucket / Addressable
# ---------------------------------------------------------------------------

class Episode(BaseModel):
    task_id: str
    code: str = ""
    diagnosis: str = ""
    error_trace: str = ""
    outcome: str = "failure"
    timestamp: float = Field(default_factory=time.time)


class SyntheticDrill(BaseModel):
    prompt: str
    expected_behavior: str
    test_code: str = ""
    rationale: str = ""


class BucketStats(BaseModel):
    hit_count: int = 0
    success_count: int = 0
    created_at: float = Field(default_factory=time.time)
    last_updated: float = Field(default_factory=time.time)


class Bucket(BaseModel):
    bucket_id: str
    title: str
    addressable_fix: str
    summary_paragraph: str = ""
    trigger_signals: List[str] = Field(default_factory=list)
    playbook: List[str] = Field(default_factory=list)
    trace_summaries: List[str] = Field(default_factory=list)
    when_to_open: str = ""
    examples: List[Episode] = Field(default_factory=list)
    synthetic_drills: List[SyntheticDrill] = Field(default_factory=list)
    stats: BucketStats = Field(default_factory=BucketStats)

    @staticmethod
    def make_id(addressable_fix: str, trigger_signals: List[str]) -> str:
        sig = f"{addressable_fix}::{'|'.join(sorted(trigger_signals))}"
        return hashlib.sha256(sig.encode()).hexdigest()[:12]


class IndexEntry(BaseModel):
    bucket_id: str
    title: str
    summary_paragraph: str = ""


class MemoryIndex(BaseModel):
    entries: List[IndexEntry] = Field(default_factory=list)
    version: int = 0


# ---------------------------------------------------------------------------
# Critic output
# ---------------------------------------------------------------------------

class MergeDecision(str, Enum):
    MERGE = "merge"
    CREATE = "create"


class CriticOutput(BaseModel):
    diagnosis: str
    addressable_fix: str
    trigger_signals: List[str]
    playbook: List[str]
    trace_summary: str = ""
    when_to_open: str = ""
    summary_paragraph: str = ""
    decision: MergeDecision
    merge_target_bucket_id: Optional[str] = None
    merge_confidence: float = 0.0
    merge_rationale: str = ""
    synthetic_drills: List[SyntheticDrill] = Field(default_factory=list)
    is_success_pattern: bool = False


# ---------------------------------------------------------------------------
# Benchmark task
# ---------------------------------------------------------------------------

class BenchmarkTask(BaseModel):
    task_id: str
    prompt: str
    entry_point: str = ""
    test_code: str = ""
    canonical_solution: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
