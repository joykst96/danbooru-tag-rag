"""
API 모듈 (배포용)

- 정적 파일 서빙: index.html(루트 UI), danbooru-tags.csv(태그 호버 설명)
- /api/generate      : 기본 모드. structured 파이프라인(core/pipeline_structured.py)을 NDJSON 스트림으로 응답
- /api/generate_split: 고급(분할) 모드. 인물칸 N + 배경칸 1 (core/pipeline_split.py)
- /api/direct_search : 단일 쿼리 벡터검색 (카테고리 지정 가능)
- /api/logs          : 인메모리 로그 조회 (디버깅)

응답 필드명은 UI(index.html)가 기대하는 형태로 매핑한다:
    ko_units / person_units / keywords / final_prompt(쉼표문자열) / suspicious_tags / english_prompt
"""

import json
import logging
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .embeddings import EmbeddingManager
from . import search as search_mod
from .pipeline_structured import stream_structured_pipeline
from .pipeline_split import stream_split_pipeline, CharBlock
from . import genlog
from .config import (
    PROJECT_ROOT, TAGS_CSV_PATH, DEFAULT_TOP_K, CATEGORY_LABELS,
    MAIN_VARIANT, GEMMA2_LOCAL_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 인덱스는 config.MAIN_VARIANT(sim) 하나 (tag-embedding 벤치 실측, config.py 참고).

# ---------------------------------------------------------------------------
# 생성 요청 직렬화 (동시 1개) + 실시간 대기열
# ---------------------------------------------------------------------------
# LLM이 GPU VRAM을 거의 다 쓰고 임베딩은 CPU라, 생성을 동시에 돌리면 경합한다.
# LLM도 단일모델이라 어차피 순차. 동시 1개로 직렬화하고 대기번호를 실시간으로 보여줘
# 사용자가 새로고침 대신 기다리게 한다. 단일 프로세스/워커라 프로세스 내 변수로 충분
# (Redis 불필요). direct_search 는 가벼우므로 큐에 안 태운다.
#
# 티켓 방식: 도착 시 번호 발급(next++). serving 이 내 번호와 같아질 때까지 대기.
# 처리 끝나면 serving 을 다음으로 넘긴다. 순서 보장 + 동시 1개 보장.
import asyncio as _asyncio
import time as _time

_gen_lock = _asyncio.Lock()           # next/serving 갱신 보호
_gen_state = {"next": 0, "serving": 1}  # 발급 카운터 / 현재 처리할 번호
_active_ips: set[str] = set()          # 큐 대기 또는 처리 중인 클라이언트 IP


def _ms_since(t: float) -> int:
    """생성 로그 timing_ms 용 경과 시간(ms)."""
    return int((_time.monotonic() - t) * 1000)


def _split_input_summary(req) -> str:
    """분할입력을 생성 로그 input 용 한 줄 요약으로. 형식은 v1 로그와 같다(기존 분석 스크립트가 이 형식을 읽는다)."""
    char_lines = []
    for c in req.characters:
        parts = []
        if c.is_original:
            parts.append("[오리지널]")
        elif c.is_passthrough:
            parts.append("[패스스루]")
            if c.series.strip():
                parts.append(f"[{c.series.strip()}]")
            if c.name.strip():
                parts.append(c.name.strip())
        else:
            if c.series.strip():
                parts.append(f"[{c.series.strip()}]")
            if c.name.strip():
                parts.append(c.name.strip())
        if c.desc.strip():
            parts.append(f"- {c.desc.strip()}")
        if parts:
            char_lines.append(" ".join(parts))
    summary = " / ".join(char_lines)
    if req.background.strip():
        summary += f" / 배경: {req.background.strip()}"
    return summary


def _client_ip(request) -> str:
    """실제 클라이언트 IP. 프록시(Cloudflare Tunnel/nginx) 뒤이므로 헤더 우선.
    CF-Connecting-IP → X-Forwarded-For(첫 IP) → request.client.host 순."""
    h = request.headers
    cf = h.get("cf-connecting-ip")
    if cf:
        return cf.strip()
    xff = h.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def _try_register_ip(ip: str) -> bool:
    """IP를 활성 집합에 등록. 이미 있으면(동일 IP 큐/처리 중) False=거절."""
    async with _gen_lock:
        if ip in _active_ips:
            return False
        _active_ips.add(ip)
        return True


async def _release_ip(ip: str) -> None:
    async with _gen_lock:
        _active_ips.discard(ip)


async def _take_ticket() -> int:
    async with _gen_lock:
        _gen_state["next"] += 1
        return _gen_state["next"]


def _waiting_count(my: int) -> int:
    """티켓 my 기준 대기 인원(자기 포함). serving 이 처리할 번호, my 가 내 번호.
    my==serving 이면 1(바로 나), 앞에 막힌 게 있으면 그만큼 +."""
    return max(1, my - _gen_state["serving"] + 1)


async def _advance_serving():
    async with _gen_lock:
        _gen_state["serving"] += 1


async def _wait_for_turn(my: int):
    """티켓 my 차례가 올 때까지 대기번호를 실시간 yield (generate/generate_split 공유)."""
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
    logger.info(f"임베딩 모델 프리로드 중... (embedding={GEMMA2_LOCAL_NAME}, variant={MAIN_VARIANT})")
    EmbeddingManager.get_model()
    try:
        search_mod.get_tagset(MAIN_VARIANT)
        search_mod.get_tag_rows(MAIN_VARIANT)   # structured 어휘 일치용 (첫 요청 지연 방지)
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
    nl_temperature: float | None = Field(
        default=None, description="자연어 온도. None 이면 톤별 권장값(분해·선택은 항상 0)"
    )
    search_categories: list[int] | None = Field(
        default=None, description="검색 풀 카테고리(고급). None/빈값이면 일반(cat0)만"
    )
    nl_tone: str | None = Field(
        default=None,
        description="자연어 톤: rich/plain. None이면 rich. 단어형 보조출력: phrase",
    )
    include_phrases: bool = Field(
        default=True,
        description="최종 태그에 DB 밖 영어 구(선택 단계가 맞는 태그를 못 찾아 낸 구)를 넣을지. 자연어에는 영향 없음",
    )


class DirectSearchRequest(BaseModel):
    query: str = Field(description="직접 검색할 단어")
    variant: str = Field(default=MAIN_VARIANT)
    top_k: int = Field(default=DEFAULT_TOP_K)
    threshold: float = Field(default=0.0)
    categories: list[int] | None = Field(default=None)


class CharBlockIn(BaseModel):
    name: str = Field(default="", description="캐릭터명(옵션)")
    series: str = Field(default="", description="작품명(옵션)")
    desc: str = Field(default="", description="캐릭터묘사(한국어)")
    is_original: bool = Field(default=False, description="오리지널 캐릭터(작품/캐릭터 검색 skip)")
    is_passthrough: bool = Field(default=False, description="패스스루(신규 캐릭터): DB 검색 skip, 입력 이름을 NL 핸들로 사용")


class GenerateSplitRequest(BaseModel):
    characters: list[CharBlockIn] = Field(default_factory=list, description="인물칸 목록")
    background: str = Field(default="", description="배경/공통요소(한국어)")
    nl_temperature: float | None = Field(
        default=None, description="자연어 온도. None 이면 톤별 권장값"
    )
    nl_tone: str | None = Field(
        default=None,
        description="자연어 톤: rich/plain. None이면 rich. 단어형 보조출력: phrase",
    )
    include_phrases: bool = Field(
        default=True,
        description="최종 태그에 DB 밖 영어 구를 넣을지(GenerateRequest 와 같음)",
    )


# ---------------------------------------------------------------------------
# /api/generate — structured 파이프라인 → NDJSON 스트림
# ---------------------------------------------------------------------------
@app.post("/api/generate")
async def generate(req: GenerateRequest, request: Request):
    client_ip = _client_ip(request)

    async def stream():
        if not req.prompt.strip():
            yield json.dumps({"status": "오류", "data": {}}, ensure_ascii=False) + "\n"
            return

        # 동일 IP가 이미 큐/처리 중이면 거절(자동화 연속요청 차단). 정상 UI는 동시요청 안 보냄.
        if not await _try_register_ip(client_ip):
            genlog.console_log_rejected("basic", client_ip)
            yield json.dumps(
                {"status": "rejected", "reason": "duplicate_ip",
                 "message": "동일 IP로 처리 중인 작업이 있습니다.", "data": {}},
                ensure_ascii=False,
            ) + "\n"
            return

        rid = genlog.new_request_id()
        t0 = _time.monotonic()
        my = await _take_ticket()
        genlog.console_log_request("basic", _waiting_count(my), req_id=rid)
        _final_tags: list[str] = []
        _nl_prompt: str = ""
        _trace: dict = {}
        _timing: dict = {"stages": {}}
        _logged = False

        def _log(status: str, err: BaseException | None = None) -> None:
            # 성공·실패·중단 모두 한 번만 파일에 남긴다. 실패·중단이면 거기까지의 단계만 담긴다.
            nonlocal _logged
            if _logged:
                return
            _logged = True
            _timing["total"] = _ms_since(t0)
            genlog.log_generation(
                user_input=req.prompt.strip(),
                final_tags=_final_tags,
                nl_prompt=_nl_prompt,
                mode="basic",
                req_id=rid,
                status=status,
                request=req.model_dump(),
                trace=_trace,
                timing_ms=_timing,
                error=err,
            )

        try:
            # 내 차례(serving == my)가 올 때까지 대기번호 실시간 표시
            async for w in _wait_for_turn(my):
                yield w
            _timing["queue"] = _ms_since(t0)
            t_turn = _time.monotonic()

            # 내 차례 — 단계별 스트리밍 (korean → english → final → nl)
            _cats = set(req.search_categories) if req.search_categories else None
            _events = stream_structured_pipeline(
                req.prompt,
                variant=MAIN_VARIANT,
                generate_nl=True,
                search_categories=_cats,
                nl_tone=req.nl_tone,
                nl_temperature=req.nl_temperature,
                include_phrases=req.include_phrases,
            )
            async for ev in _events:
                stage = ev["stage"]
                d = ev["data"]
                _timing["stages"][stage] = _ms_since(t_turn)
                out: dict = {}
                if stage == "korean":
                    out["ko_units"] = d["ko_units"]
                    out["person_units"] = d["person_units"]
                    _trace["ko_units"] = d["ko_units"]
                elif stage == "english":
                    out["keywords"] = d["en_units"]
                    _trace["en_units"] = d["en_units"]
                elif stage == "final":
                    out["final_prompt"] = ", ".join(d["final_tags"])
                    out["suspicious_tags"] = d["hallucinated"]
                    _final_tags = d["final_tags"]          # 원본(언더스코어 포함) — genlog가 치환
                    _trace["hallucinated"] = d["hallucinated"]
                    _trace.update(d.get("trace") or {})    # 파일 로그 전용(클라이언트로 안 보냄)
                elif stage == "nl":
                    out["english_prompt"] = d["nl_prompt"]
                    _nl_prompt = d["nl_prompt"]

                yield json.dumps(
                    {"status": ev["status"], "data": out}, ensure_ascii=False
                ) + "\n"

            _log("ok")
            genlog.console_log_generation(
                user_input=req.prompt.strip(),
                final_tags=_final_tags,
                nl_prompt=_nl_prompt,
                mode="basic",
                settings={
                    "톤": req.nl_tone or "rich",
                    "검색범위": req.search_categories or "일반(기본)",
                    "자연어온도": req.nl_temperature if req.nl_temperature is not None else "톤 기본",
                },
                req_id=rid,
            )
        except Exception as e:
            # 파이프라인 예외 — 콘솔과 파일(status=error)에 남기고 차례는 넘긴다.
            genlog.console_log_error("basic", req.prompt.strip(), e, req_id=rid)
            _log("error", e)
            raise
        finally:
            # 여기까지 기록이 없으면 클라이언트 연결끊김 등으로 중단된 것(GeneratorExit/CancelledError).
            _log("aborted")
            # 성공/실패/연결끊김 무엇이든 다음 사람에게 차례를 넘겨야 큐가 안 막힘
            await _advance_serving()
            await _release_ip(client_ip)

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )


@app.post("/api/generate_split")
async def generate_split(req: GenerateSplitRequest, request: Request):
    """분할입력(고급) — 인물칸 N + 배경칸 1. 큐는 /api/generate 와 공유."""
    client_ip = _client_ip(request)

    async def stream():
        blocks = [
            CharBlock(name=c.name, series=c.series, desc=c.desc, is_original=c.is_original, is_passthrough=c.is_passthrough)
            for c in req.characters
        ]
        has_char = any(
            b.name.strip() or b.series.strip() or b.desc.strip() for b in blocks
        )
        if not has_char and not req.background.strip():
            yield json.dumps({"status": "오류", "data": {}}, ensure_ascii=False) + "\n"
            return

        # 동일 IP 큐/처리 중이면 거절(자동화 차단). 큐는 generate 와 공유하므로 IP 집합도 공유.
        if not await _try_register_ip(client_ip):
            genlog.console_log_rejected("split", client_ip)
            yield json.dumps(
                {"status": "rejected", "reason": "duplicate_ip",
                 "message": "동일 IP로 처리 중인 작업이 있습니다.", "data": {}},
                ensure_ascii=False,
            ) + "\n"
            return

        rid = genlog.new_request_id()
        t0 = _time.monotonic()
        my = await _take_ticket()
        genlog.console_log_request("split", _waiting_count(my), req_id=rid)
        _final_tags: list[str] = []
        _nl_prompt: str = ""
        _err_input: str = f"{len(req.characters)}인물" + (" +배경" if req.background.strip() else "")
        _trace: dict = {"characters": []}
        _timing: dict = {"stages": {}}
        _logged = False
        input_summary = _split_input_summary(req)

        def _log(status: str, err: BaseException | None = None) -> None:
            # 성공·실패·중단 모두 한 번만 파일에 남긴다. 분할입력은 사람이 읽기 좋은 요약을 input 에.
            nonlocal _logged
            if _logged:
                return
            _logged = True
            _timing["total"] = _ms_since(t0)
            genlog.log_generation(
                user_input=input_summary,
                final_tags=_final_tags,
                nl_prompt=_nl_prompt,
                mode="split",
                extra={   # v1 과 같은 자리·형식(기존 분석 스크립트 호환). 전체 요청은 request 에.
                    "characters": [
                        {"name": c.name, "series": c.series, "desc": c.desc, "is_passthrough": c.is_passthrough}
                        for c in req.characters
                    ],
                    "background": req.background,
                },
                req_id=rid,
                status=status,
                request=req.model_dump(),
                trace=_trace,
                timing_ms=_timing,
                error=err,
            )

        try:
            async for w in _wait_for_turn(my):
                yield w
            _timing["queue"] = _ms_since(t0)
            t_turn = _time.monotonic()

            async for ev in stream_split_pipeline(
                blocks,
                background_desc=req.background,
                variant=MAIN_VARIANT,
                top_k=5,
                generate_nl=True,
                nl_tone=req.nl_tone,
                nl_temperature=req.nl_temperature,
                include_phrases=req.include_phrases,
            ):
                stage = ev["stage"]
                d = ev["data"]
                _timing["stages"][stage if stage != "character" else f"character{d.get('index')}"] = _ms_since(t_turn)
                out: dict = {}
                if stage == "character":
                    out["character"] = d
                    _trace["characters"].append(d)
                elif stage == "background":
                    out["background_tags"] = d["background_tags"]
                    _trace["background_tags"] = d["background_tags"]
                elif stage == "final":
                    out["final_prompt"] = ", ".join(d["final_tags"])
                    _final_tags = d["final_tags"]          # 원본(언더스코어 포함) — genlog가 치환
                    if d.get("phrases_dropped"):
                        _trace["phrases_dropped"] = d["phrases_dropped"]
                elif stage == "nl":
                    out["english_prompt"] = d["nl_prompt"]
                    _nl_prompt = d["nl_prompt"]

                yield json.dumps(
                    {"status": ev["status"], "data": out}, ensure_ascii=False
                ) + "\n"

            _log("ok")
            genlog.console_log_generation(
                user_input=input_summary,
                final_tags=_final_tags,
                nl_prompt=_nl_prompt,
                mode="split",
                settings={
                    "톤": req.nl_tone or "rich",
                    "자연어온도": req.nl_temperature if req.nl_temperature is not None else "톤 기본",
                    "인물수": len(req.characters),
                },
                req_id=rid,
            )
        except Exception as e:
            genlog.console_log_error("split", _err_input, e, req_id=rid)
            _log("error", e)
            raise
        finally:
            _log("aborted")   # 기록이 없으면 연결끊김 등으로 중단된 것
            await _advance_serving()
            await _release_ip(client_ip)

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
    cats = "+".join(CATEGORY_LABELS.get(c, str(c)) for c in sorted(include)) if include else "전체"
    logger.info(f'🔎 벡터DB 직접조회: [{cats}] "{req.query}" (threshold={req.threshold})')
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
    logger.info(f"   \u2514 {len(results)}건 반환")
    return {"query": req.query, "variant": req.variant, "results": results}


@app.get("/api/logs")
async def get_logs(after: int = 0):
    items = [x for x in _LOG_BUFFER if x["seq"] > after]
    return {"logs": items, "last": _LOG_SEQ["n"]}


@app.get("/api/health")
async def health():
    return {"status": "ok", "variant": MAIN_VARIANT, "embedding": GEMMA2_LOCAL_NAME}


# ---------------------------------------------------------------------------
# 정적 파일 서빙
# ---------------------------------------------------------------------------
@app.get("/")
async def index():
    # index.html 은 절대 캐시하지 않는다(인라인 JS/CSS 포함 단일 파일).
    # 업데이트 시 클라이언트가 강제 새로고침 없이 바로 최신 UI 를 받게 한다.
    return FileResponse(
        PROJECT_ROOT / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# 프론트엔드 정적 파일(web/: CSS·JS 모듈). index.html 과 함께 배포하므로 매번 변경 확인(no-cache, ETag 304).
# 확장자별 타입을 직접 정한다: Windows mimetypes 는 레지스트리 영향으로 .js 를 text/plain 으로 줄 수 있고,
# 그러면 브라우저가 <script type="module"> 을 거부한다.
WEB_DIR = (PROJECT_ROOT / "web").resolve()
_WEB_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json",
}


@app.get("/web/{path:path}")
async def web_asset(path: str):
    target = (WEB_DIR / path).resolve()
    if not target.is_relative_to(WEB_DIR) or not target.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(
        target,
        media_type=_WEB_TYPES.get(target.suffix.lower()),
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/danbooru-tags.csv")
async def tags_csv():
    if TAGS_CSV_PATH.exists():
        # CSV(태그 호버 설명)는 큰 파일이라 캐시하되, 인덱스 갱신 시 바뀔 수 있으므로
        # ETag 재검증을 강제한다. FileResponse 가 ETag/Last-Modified 를 자동 부여하므로
        # no-cache(=캐시하되 매번 변경 확인)로 묵은 파일 방지 + 미변경 시 304 로 대역폭 절약.
        return FileResponse(
            TAGS_CSV_PATH,
            media_type="text/csv",
            headers={"Cache-Control": "no-cache"},
        )
    return JSONResponse({"error": "csv not found"}, status_code=404)


def main():
    import uvicorn
    uvicorn.run("core.api:app", host="0.0.0.0", port=3333, reload=False)


if __name__ == "__main__":
    main()
