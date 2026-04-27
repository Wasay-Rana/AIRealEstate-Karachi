from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional


@dataclass
class IngestJob:
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    chunks_created: Optional[int] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class IngestStatusTracker:
    def __init__(self) -> None:
        self._jobs: dict[str, IngestJob] = {}

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = IngestJob(job_id=job_id, status="queued")
        return job_id

    def update_job(
        self,
        job_id: str,
        status: Literal["queued", "processing", "completed", "failed"],
        chunks_created: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        if job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        job.status = status
        job.updated_at = datetime.utcnow()
        if chunks_created is not None:
            job.chunks_created = chunks_created
        if error is not None:
            job.error = error

    def get_job(self, job_id: str) -> Optional[IngestJob]:
        return self._jobs.get(job_id)
