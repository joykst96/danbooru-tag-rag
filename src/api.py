"""
API 모듈 (얇은 FastAPI 계층)

HTTP 요청을 받아 pipeline 을 호출하고 결과를 SSE(JSON 라인)로 스트리밍한다.
실제 로직(검색/LLM/2-pass)은 모두 search/llm/pipeline 모듈에 있으며,
이 파일은 입출력과 스트리밍만 담당한다.
"""

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .embeddings import EmbeddingManager
from .pipeline import run_pipeline, PipelineConfig, PipelineResult
from . import search as search_mod
from .config import DEFAULT_VARIANT, DEFAULT_TOP_K, DEFAULT_THRESHOLD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 임베딩 모델 프리로드 (콜드스타트 방지)."""
    logger.info("임베딩 모델 프리로드 중...")
    EmbeddingManager.get_model()
    logger.info("준비 완료")
    yield


app = FastAPI(title="Danbooru RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 요청 스키마
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field(description="한국어 프롬프트")
    pass1_variant: str = Field(default=DEFAULT_VARIANT)
    pass2_variant: str = Field(default=DEFAULT_VARIANT)
    use_pass1: bool = Field(default=True)
    generate_nl: bool = Field(default=True)


class DirectSearchRequest(BaseModel):
    query: str = Field(description="DB에 직접 검색할 단어/문장")
    variant: str = Field(default=DEFAULT_VARIANT)
    top_k: int = Field(default=DEFAULT_TOP_K)
    threshold: float = Field(default=0.0, description="이 점수 미만 결과는 버림")


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------
@app.post("/generate")
async def generate(req: GenerateRequest):
    """2-pass 파이프라인 실행 후 단계별 진행상황을 SSE(JSON 라인)로 스트리밍."""
    cfg = PipelineConfig(
        pass1_variant=req.pass1_variant,
        pass2_variant=req.pass2_variant,
        use_pass1=req.use_pass1,
        generate_nl=req.generate_nl,
    )

    async def stream():
        # 단계 콜백: 진행상황을 JSON 라인으로 흘려보냄
        queue: list[str] = []

        async def cb(step: int, status: str, data: dict | None):
            queue.append(json.dumps(
                {"step": step, "status": status, "data": data},
                ensure_ascii=False,
            ) + "\n")

        # run_pipeline 을 태스크로 돌리면서 콜백 큐를 흘리는 대신,
        # 여기서는 단순화를 위해 콜백이 큐에 쌓은 걸 파이프라인 종료 후 함께 전송한다.
        # (실시간성이 더 필요하면 asyncio.Queue 로 교체 가능 — 아래 주석 참고)
        result: PipelineResult = await run_pipeline(req.prompt, cfg, cb)

        # 누적된 단계 메시지 전송
        for line in queue:
            yield line

        # 최종 결과 전송
        final_prompt = ", ".join(result.final_tags)
        yield json.dumps({
            "step": 99,
            "status": "완료",
            "data": {
                "final_tags": result.final_tags,
                "final_prompt": final_prompt,
                "grouped": result.grouped,
                "nl_prompt": result.nl_prompt,
                "suspicious": result.suspicious,
                "keywords": result.keywords,
                "refined": result.refined,
            },
        }, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/direct_search")
async def direct_search(req: DirectSearchRequest):
    """LLM 없이 단일 쿼리를 지정 variant DB에서 검색 (디버깅/프론트 직접조회용)."""
    hits = search_mod.search_one(req.query, variant=req.variant, top_k=req.top_k)
    results = [
        {
            "tag": r.tag,
            "score": r.score,
            "category": r.category,
            "major": r.major,
            "minor": r.minor,
            "aliases": r.aliases,
        }
        for r in hits.results
        if r.score >= req.threshold
    ]
    return {"query": req.query, "variant": req.variant, "results": results}


def main():
    import uvicorn
    uvicorn.run("danbooru_rag.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
