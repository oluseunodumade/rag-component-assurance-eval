"""FastAPI service for RAG component assurance evaluation.

Run from the backend directory:

    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import Body, FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.dataset import load_dataset
from app.llm import LLMCallError, LLMNotConfiguredError
from app.pipeline import run_evaluation
from app.playground import apply_baseline, apply_runtime_controls, build_custom_item
from app.schemas import AuditItem, AuditItemResult, BatchAuditReport, EvaluateRequest, HealthResponse

LOCAL_ORIGIN = "http://localhost:3000"


def create_app() -> FastAPI:
    """Build the API with the sample dataset loaded once."""
    dataset = load_dataset()
    by_id = {item.query_id: item for item in dataset}
    app = FastAPI(
        title="RAG Component Assurance Eval",
        version="0.1.0",
        description=(
            "Audits query rewriting, routing, retrieval, and response generation "
            "for a logged RAG dataset. mock_mode=true runs entirely offline."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[LOCAL_ORIGIN],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(LLMNotConfiguredError)
    async def _unconfigured(_request: object, exc: LLMNotConfiguredError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(LLMCallError)
    async def _llm_failed(_request: object, exc: LLMCallError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.get("/api/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        settings = get_settings()
        return HealthResponse(
            status="ok",
            llm_configured=settings.llm_configured,
            dataset_size=len(dataset),
            mock_mode_default=True,
        )

    @app.get("/api/dataset", response_model=list[AuditItem], tags=["dataset"])
    def get_dataset() -> list[AuditItem]:
        return dataset

    @app.post("/api/evaluate", response_model=BatchAuditReport, tags=["evaluation"])
    def evaluate(
        payload: EvaluateRequest = Body(default_factory=EvaluateRequest),
    ) -> BatchAuditReport:
        """Run the audit on the dataset, one logged query, or a custom playground query."""
        items = _resolve_items(payload, dataset, by_id)
        return run_evaluation(items, mock_mode=payload.mock_mode)

    @app.post(
        "/api/evaluate/{query_id}",
        response_model=AuditItemResult,
        tags=["evaluation"],
    )
    def evaluate_one(
        query_id: str = Path(min_length=1),
        payload: EvaluateRequest = Body(default_factory=EvaluateRequest),
    ) -> AuditItemResult:
        """Run the four-stage audit for a single dataset query."""
        if query_id not in by_id:
            raise HTTPException(status_code=404, detail=f"Query '{query_id}' was not found in the audit dataset.")
        scoped = payload.model_copy(update={"query_id": query_id, "custom_query": None, "experiment": None})
        report = run_evaluation(_resolve_items(scoped, dataset, by_id), mock_mode=payload.mock_mode)
        return report.items[0]

    return app


def _resolve_items(
    payload: EvaluateRequest,
    dataset: list[AuditItem],
    by_id: dict[str, AuditItem],
) -> list[AuditItem]:
    custom = (payload.custom_query or "").strip()
    if custom:
        return [
            build_custom_item(
                custom,
                dataset,
                top_k=payload.top_k,
                prompt_version=payload.prompt_version,
                mock_mode=payload.mock_mode,
            )
        ]
    if payload.query_id:
        item = by_id.get(payload.query_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"Query '{payload.query_id}' was not found in the audit dataset.")
        if payload.experiment == "baseline":
            return [apply_baseline(item)]
        if payload.experiment == "candidate":
            return [item]
        return [apply_runtime_controls(item, top_k=payload.top_k, prompt_version=payload.prompt_version)]
    if payload.experiment == "baseline":
        return [apply_baseline(item) for item in dataset]
    return list(dataset)


app = create_app()
