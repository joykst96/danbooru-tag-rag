"""
API 모듈 (배포용)

- 정적 파일 서빙: index.html(루트 UI), danbooru-tags.csv(태그 호버 설명)
- /api/generate     : 4회 분해 파이프라인을 구버전 호환 SSE 형식으로 응답
- /api/direct_search: 단일 쿼리 벡터검색 (카테고리 지정 가능)
- /api/logs         : 인메모리 로그 조회 (디버깅)

내부 로직은 4회 분해 파이프라인(pipeline_decomposed)을 본선으로 사용.
응답 필드명만 구버전 UI(index.html)가 기대하는 형태로 매핑:
    final_prompt(쉼표문자열) / english_prompt / suspicious_tags / keywords / candidates
"""

import json
import logging
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .embeddings import EmbeddingManager
from . import search as search_mod
from .pipeline_decomposed import run_decomposed_pipeline, stream_decomposed_pipeline
from .config import PROJECT_ROOT, TAGS_CSV_PATH, DEFAULT_TOP_K

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 본선 variant 확정: B (실측 — 코드스위칭 분포서 영어태그명 포함 B가 최적)
MAIN_VARIANT = "b"

# ---------------------------------------------------------------------------
# 생성 요청 직렬화 (동시 1개) + 실시간 대기열
# ---------------------------------------------------------------------------
# 4060이 LLM으로 VRAM 꽉 차고 임베딩도 CPU라 생성을 동시에 돌리면 경합.
# LLM도 단일모델이라 어차피 순차. 동시 1개로 직렬화하고 대기번호를 실시간으로 보여줘
# 사용자가 새로고침 대신 기다리게 한다. 단일 프로세스/워커라 프로세스 내 변수로 충분
# (Redis 불필요). direct_search 는 가벼우므로 큐에 안 태운다.
#
# 티켓 방식: 도착 시 번호 발급(next++). serving 이 내 번호와 같아질 때까지 대기.
# 처리 끝나면 serving 을 다음으로 넘긴다. 순서 보장 + 동시 1개 보장.
import asyncio as _asyncio

_gen_lock = _asyncio.Lock()           # next/serving 갱신 보호
_gen_state = {"next": 0, "serving": 1}  # 발급 카운터 / 현재 처리할 번호


async def _take_ticket() -> int:
    async with _gen_lock:
        _gen_state["next"] += 1
        return _gen_state["next"]


async def _advance_serving():
    async with _gen_lock:
        _gen_state["serving"] += 1


# ---------------------------------------------------------------------------
# 인메모리 로그 캡처
# ---------------------------------------------------------------------------
_LOG_BUFFER: deque = deque(maxlen=500)
_LOG_SEQ = {"n": 0}


class BufferLogHandler(logging.Handler):
    def emit(self, record):
        try:
            _LOG_SEQ["n"] += 1
            _LOG_BUFFER.append({
                "seq": _LOG_SEQ["n"],
                "level": record.levelname,
                "name": record.name.replace("core.", ""),
                "msg": self.format(record),
            })
        except Exception:
            pass


_buf_handler = BufferLogHandler()
_buf_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger("core").addHandler(_buf_handler)
logging.getLogger("core").setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("임베딩 모델 프리로드 중...")
    EmbeddingManager.get_model()
    try:
        search_mod.get_tagset(MAIN_VARIANT)
    except Exception as e:
        logger.warning(f"태그집합 프리로드 실패(첫 요청 시 재시도): {e}")
    logger.info("준비 완료")
    yield


app = FastAPI(title="Danbooru RAG", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 요청 스키마
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field(description="한국어 프롬프트")
    threshold: float = Field(default=0.0)
    kw_temperature: float = Field(default=0.1)
    nl_temperature: float = Field(default=0.4)


class DirectSearchRequest(BaseModel):
    query: str = Field(description="직접 검색할 단어")
    variant: str = Field(default=MAIN_VARIANT)
    top_k: int = Field(default=DEFAULT_TOP_K)
    threshold: float = Field(default=0.0)
    categories: list[int] | None = Field(default=None)


# ---------------------------------------------------------------------------
# /api/generate — 4회 분해 → 구버전 호환 SSE
# ---------------------------------------------------------------------------
@app.post("/api/generate")
async def generate(req: GenerateRequest):
    async def stream():
        if not req.prompt.strip():
            yield json.dumps({"status": "오류", "data": {}}, ensure_ascii=False) + "\n"
            return

        my = await _take_ticket()
        try:
            # 내 차례(serving == my)가 올 때까지 대기번호 실시간 표시
            notified = None
            while _gen_state["serving"] < my:
                ahead = my - _gen_state["serving"]
                if ahead != notified:
                    notified = ahead
                    yield json.dumps(
                        {"status": f"대기 중... 앞에 {ahead}명", "data": {}},
                        ensure_ascii=False,
                    ) + "\n"
                await _asyncio.sleep(0.8)

            # 내 차례 — 4스텝 스트리밍
            async for ev in stream_decomposed_pipeline(
                req.prompt,
                general_variant=MAIN_VARIANT,
                en_variant=MAIN_VARIANT,
                top_k=5,
                generate_nl=True,
            ):
                stage = ev["stage"]
                d = ev["data"]
                out: dict = {}
                if stage == "korean":
                    out["ko_units"] = d["ko_units"]
                    out["person_units"] = d["person_units"]
                elif stage == "english":
                    out["keywords"] = d["en_units"]
                elif stage == "final":
                    out["final_prompt"] = ", ".join(d["final_tags"])
                    out["suspicious_tags"] = d["hallucinated"]
                elif stage == "nl":
                    out["english_prompt"] = d["nl_prompt"]

                yield json.dumps(
                    {"status": ev["status"], "data": out}, ensure_ascii=False
                ) + "\n"
        finally:
            # 성공/실패/연결끊김 무엇이든 다음 사람에게 차례를 넘겨야 큐가 안 막힘
            await _advance_serving()

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )


@app.post("/api/direct_search")
async def direct_search(req: DirectSearchRequest):
    include = set(req.categories) if req.categories else None
    hits = search_mod.search_one(
        req.query, variant=req.variant, top_k=req.top_k, include_categories=include
    )
    results = [
        {
            "tag": r.tag, "score": r.score, "category": r.category,
            "major": r.major, "minor": r.minor,
            "definition": r.definition, "aliases": r.aliases,
        }
        for r in hits.results if r.score >= req.threshold
    ]
    return {"query": req.query, "variant": req.variant, "results": results}


@app.get("/api/logs")
async def get_logs(after: int = 0):
    items = [x for x in _LOG_BUFFER if x["seq"] > after]
    return {"logs": items, "last": _LOG_SEQ["n"]}


@app.get("/api/health")
async def health():
    return {"status": "ok", "variant": MAIN_VARIANT}


# ---------------------------------------------------------------------------
# 정적 파일 서빙
# ---------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(PROJECT_ROOT / "index.html")


@app.get("/danbooru-tags.csv")
async def tags_csv():
    if TAGS_CSV_PATH.exists():
        return FileResponse(TAGS_CSV_PATH, media_type="text/csv")
    return JSONResponse({"error": "csv not found"}, status_code=404)


def main():
    import uvicorn
    uvicorn.run("core.api:app", host="0.0.0.0", port=3333, reload=False)


if __name__ == "__main__":
    main()
