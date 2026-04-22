"""
CS4241 — Introduction to Artificial Intelligence
Student: Farima Konaré | Index: 10012200004
Module: logger.py — Structured per-stage pipeline logger.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class StageLog:
    stage: str
    timestamp: float
    duration_ms: float
    data: Dict[str, Any]


@dataclass
class PipelineLog:
    query: str
    stages: List[StageLog] = field(default_factory=list)
    total_duration_ms: float = 0.0
    _start: float = field(default_factory=time.time, repr=False)

    def log(self, stage: str, data: Dict[str, Any], duration_ms: float = 0.0) -> None:
        self.stages.append(StageLog(
            stage=stage,
            timestamp=time.time(),
            duration_ms=round(duration_ms, 2),
            data=data,
        ))

    def finalize(self) -> None:
        self.total_duration_ms = round((time.time() - self._start) * 1000, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "total_duration_ms": self.total_duration_ms,
            "stages": [
                {"stage": s.stage, "duration_ms": s.duration_ms, "data": s.data}
                for s in self.stages
            ],
        }


class PipelineLogger:
    """Measures per-stage timing for a single pipeline run."""

    def __init__(self, query: str):
        self.log = PipelineLog(query=query)
        self._stage_start = time.time()

    def begin_stage(self) -> None:
        self._stage_start = time.time()

    def end_stage(self, stage: str, data: Dict[str, Any]) -> None:
        duration_ms = (time.time() - self._stage_start) * 1000
        self.log.log(stage, data, duration_ms)
        self.begin_stage()

    def done(self) -> PipelineLog:
        self.log.finalize()
        return self.log
