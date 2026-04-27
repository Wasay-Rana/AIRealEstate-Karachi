from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.core.dependencies import (
    get_embedder,
    get_lightrag_store,
    get_pinecone_store,
    get_status_tracker,
)
from app.ingest.pipeline import IngestPipeline
from app.models.requests import IngestRequest
from app.models.responses import IngestResponse

router = APIRouter(prefix="/api/v1", tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest_document(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    tracker = get_status_tracker()
    job_id = tracker.create_job()

    pipeline = IngestPipeline(
        embedder=get_embedder(),
        pinecone_store=get_pinecone_store(),
        lightrag_store=get_lightrag_store(),
        status_tracker=tracker,
    )

    if request.background:
        background_tasks.add_task(pipeline.run, request, job_id)
        return IngestResponse(
            job_id=job_id,
            status="queued",
            message="Ingestion queued. Poll /api/v1/ingest/{job_id} for status.",
        )

    await pipeline.run(request, job_id)
    job = tracker.get_job(job_id)
    return IngestResponse(
        job_id=job_id,
        status=job.status,  # type: ignore[union-attr]
        chunks_created=job.chunks_created,  # type: ignore[union-attr]
        message="Ingestion complete" if job.status == "completed" else job.error or "Failed",  # type: ignore[union-attr]
    )


@router.get("/ingest/{job_id}", response_model=IngestResponse)
async def get_ingest_status(job_id: str) -> IngestResponse:
    tracker = get_status_tracker()
    job = tracker.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return IngestResponse(
        job_id=job.job_id,
        status=job.status,
        chunks_created=job.chunks_created,
        message=job.error or job.status,
    )
