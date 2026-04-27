from fastapi import Request
from fastapi.responses import JSONResponse


class IngestError(Exception):
    def __init__(self, message: str, job_id: str = ""):
        self.message = message
        self.job_id = job_id
        super().__init__(message)


class RetrievalError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class GenerationError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class StorageError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


async def ingest_error_handler(request: Request, exc: IngestError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": "IngestError", "message": exc.message, "job_id": exc.job_id},
    )


async def retrieval_error_handler(request: Request, exc: RetrievalError) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "RetrievalError", "message": exc.message},
    )


async def generation_error_handler(request: Request, exc: GenerationError) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "GenerationError", "message": exc.message},
    )
