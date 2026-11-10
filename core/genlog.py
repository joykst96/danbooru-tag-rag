"""
생성 로그 파일 기록 모듈

생성 1건을 파일에 한 줄(JSON Lines)로 남긴다. 나중에 같은 조건으로 재현하고, 어느 단계에서
틀렸는지 로그만 보고 가를 수 있게 하는 것이 목적이다.
최종 태그는 언더스코어(_)를 공백으로 치환해 저장한다(사용자 요구). 원본은 final_tags_raw 에 따로 둔다.

- 경로: 환경변수 GEN_LOG_PATH (기본 /app/logs/generations.jsonl).
  Docker compose 에서 ./logs:/app/logs 볼륨으로 호스트에 보존.
- 단일 워커 전제(api 와 동일). 프로세스 내 단순 append.
- 자정(로컬) 기준 일자별 파일 분리(generations-YYYY-MM-DD.jsonl)로 무한 증식 방지.
- 기록 실패가 응답을 막지 않도록 모든 예외를 삼킨다(로그는 보조 기능).

레코드 형식 (v2, 2026-11-11). v1 키는 이름·의미를 그대로 두고 키만 덧붙였다(기존 분석 스크립트 호환).
  v1 키: ts, mode, input, final_tags(공백 치환), final_tags_str, nl_prompt, extra(split 칸 구조)
  v2 추가:
    v              형식 버전(2). v 가 없으면 v1.
    id             요청 id(12자리 hex). 콘솔 로그에도 같은 id 를 찍는다.
    status         ok | error | aborted (v1 은 ok 만 기록했다)
    final_tags_raw 언더스코어 원본 태그
    request        요청 본문 전체(톤·검색범위·자연어 온도, split 의 is_original 등 v1 에 없던 설정 포함)
    trace          단계별 중간 결과(분해 조각·후보·선택, 칸별 결과 등). 실패·중단이면 거기까지만.
    timing_ms      queue(대기), stages(차례가 온 뒤 단계별 누적), total
    error          {type, message} (status=error 일 때)
    build          코드·데이터 버전(commit, code_sha, tags_csv sha·행 수, 인덱스, 모델). 프로세스당 한 번 계산.
  ts 는 v2 부터 UTC 오프셋을 붙인다(예: 2026-11-11T10:00:00+09:00).
"""

import os
import json
import uuid
import hashlib
import logging
import subprocess
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# 기본 경로(컨테이너): /app/logs/. 환경변수로 덮어쓰기 가능.
_LOG_PATH = Path(os.environ.get("GEN_LOG_PATH", "/app/logs/generations.jsonl"))

SCHEMA_VERSION = 2


def new_request_id() -> str:
    """요청 id. 파일 로그와 콘솔 로그를 잇는 데 쓴다."""
    return uuid.uuid4().hex[:12]


# ── 빌드 정보 (프로세스당 한 번) ───────────────────────────────────
_BUILD: dict | None = None


def _sha256_short(data_iter) -> str:
    h = hashlib.sha256()
    for chunk in data_iter:
        h.update(chunk)
    return h.hexdigest()[:12]


def _file_chunks(path: Path, size: int = 1 << 20):
    with open(path, "rb") as f:
        while chunk := f.read(size):
            yield chunk


def _code_sha(root: Path) -> str:
    """core/*.py 내용 해시. 이미지 안에는 .git 이 없어서 commit 대신 코드 동일성 판단에 쓴다."""
    files = sorted((root / "core").glob("*.py"))
    return _sha256_short(
        p.name.encode() + b"\0" + p.read_bytes().replace(b"\r\n", b"\n") for p in files
    )


def _git_commit(root: Path) -> str | None:
    """GIT_COMMIT 환경변수(이미지 빌드 인자) → 없으면 git 작업 트리에서 읽는다. 수정 중이면 +dirty."""
    env = os.environ.get("GIT_COMMIT", "").strip()
    if env and env != "unknown":
        return env
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5)
        if head.returncode != 0:
            return None
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "core"],
                               capture_output=True, text=True, timeout=5)
        return head.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


def build_info() -> dict:
    """코드·데이터 버전. 첫 호출 때 계산해 캐시한다(태그 CSV 해시 포함, 수십 ms)."""
    global _BUILD
    if _BUILD is not None:
        return _BUILD
    info: dict = {}
    try:
        from .config import (PROJECT_ROOT, TAGS_CSV_PATH, MAIN_VARIANT, GEMMA2_LOCAL_NAME,
                             LLM_MODEL, get_lancedb_path)
        info["commit"] = _git_commit(PROJECT_ROOT)
        info["code_sha"] = _code_sha(PROJECT_ROOT)
        if TAGS_CSV_PATH.exists():
            with open(TAGS_CSV_PATH, "rb") as f:
                rows = sum(1 for _ in f) - 1          # 헤더 제외
            info["tags_csv"] = {"sha": _sha256_short(_file_chunks(TAGS_CSV_PATH)), "rows": rows}
        info["index"] = get_lancedb_path(MAIN_VARIANT).name
        info["embed_model"] = GEMMA2_LOCAL_NAME
        info["llm_model"] = LLM_MODEL
        info["started"] = datetime.now().astimezone().isoformat(timespec="seconds")
    except Exception as e:
        logger.warning(f"빌드 정보 수집 실패(로그에는 있는 것만 남김): {e}")
    _BUILD = info
    return info


def _dated_path() -> Path:
    """generations.jsonl → generations-YYYY-MM-DD.jsonl (일자별 분리)."""
    day = datetime.now().strftime("%Y-%m-%d")
    stem = _LOG_PATH.stem          # "generations"
    suffix = _LOG_PATH.suffix      # ".jsonl"
    return _LOG_PATH.with_name(f"{stem}-{day}{suffix}")


def _spacify(tags: list[str]) -> list[str]:
    """태그의 언더스코어를 공백으로 치환(저장용)."""
    return [t.replace("_", " ") for t in tags]


def log_generation(
    user_input: str,
    final_tags: list[str],
    nl_prompt: str,
    mode: str = "basic",
    extra: dict | None = None,
    *,
    req_id: str | None = None,
    status: str = "ok",
    request: dict | None = None,
    trace: dict | None = None,
    timing_ms: dict | None = None,
    error: BaseException | None = None,
) -> None:
    """
    생성 1건을 파일에 기록. 형식은 모듈 docstring(v2) 참고.

    Args:
        user_input: 사용자가 넣은 원본 입력(기본=한 문장, 분할=구조 요약 문자열).
        final_tags: 최종 태그 리스트(언더스코어 포함된 원본). final_tags 는 공백 치환본, final_tags_raw 는 원본.
        nl_prompt: 최종 자연어 프롬프트.
        mode: "basic" | "split".
        extra: 추가로 남길 메타(선택). v1 과 같은 자리.
        req_id / status / request / trace / timing_ms / error: v2 필드.
    """
    try:
        path = _dated_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "v": SCHEMA_VERSION,
            "id": req_id or new_request_id(),
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": status,
            "mode": mode,
            "input": user_input,
            "final_tags": _spacify(final_tags),                # 언더스코어→공백
            "final_tags_str": ", ".join(_spacify(final_tags)),  # 사람이 보기 쉬운 한 줄
            "nl_prompt": nl_prompt,
        }
        if extra:
            record["extra"] = extra
        record["final_tags_raw"] = list(final_tags)
        if request is not None:
            record["request"] = request
        if trace:
            record["trace"] = trace
        if timing_ms:
            record["timing_ms"] = timing_ms
        if error is not None:
            record["error"] = {"type": type(error).__name__, "message": str(error)[:500]}
        record["build"] = build_info()

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        # 로그 실패는 응답을 막지 않는다.
        logger.warning(f"생성 로그 기록 실패: {e}")


# ── 콘솔(stdout) 로그 ──────────────────────────────────────────────
# 파일 기록(log_generation)과 별개로, 실시간 모니터링용 콘솔 출력.
# docker compose logs -f 로 바로 확인. mode 이모지로 한눈에 구분.

_MODE_EMOJI = {"basic": "🟢 일반 모드", "split": "🔵 고급 모드"}


def _rid(req_id: str | None) -> str:
    """콘솔 줄에 붙이는 요청 id 표시. 파일 로그의 id 와 같다."""
    return f" #{req_id}" if req_id else ""


def console_log_generation(
    user_input: str,
    final_tags: list[str],
    nl_prompt: str,
    mode: str = "basic",
    settings: dict | None = None,
    req_id: str | None = None,
) -> None:
    """
    생성 1건을 콘솔에 보기 좋게 출력(파일 기록과 독립).

    settings: 생성에 쓰인 설정(톤/검색범위/temperature 등)을 한 줄로 덧붙인다.
    """
    try:
        head = _MODE_EMOJI.get(mode, f"⚪ {mode}")
        tags_str = ", ".join(_spacify(final_tags))
        logger.info(f"{head}{_rid(req_id)}")
        logger.info(f"📝 사용자 프롬프트: {user_input}")
        if settings:
            logger.info(f"⚙️  설정: {_format_settings(settings)}")
        logger.info(f"✨ 최종 반환 프롬프트: {tags_str}")
        if nl_prompt:
            logger.info(f"   └ 자연어: {nl_prompt}")
    except Exception as e:
        logger.warning(f"콘솔 로그 출력 실패: {e}")


def _format_settings(settings: dict) -> str:
    """설정 dict 를 사람이 읽기 좋은 한 줄로. None/빈값은 생략."""
    parts = []
    for k, v in settings.items():
        if v is None or v == "":
            continue
        parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else "(기본값)"


# ── 요청 수신 / 에러 콘솔 로그 ─────────────────────────────────────
def console_log_request(mode: str, waiting: int, req_id: str | None = None) -> None:
    """
    요청이 들어온 즉시 콘솔에 한 줄 남긴다(입력 내용 dump 없이 간단히).

    waiting: 이 요청 포함 대기 인원(자기 차례까지 앞에 몇 명+자기). 1이면 바로 처리.
    """
    try:
        head = _MODE_EMOJI.get(mode, f"⚪ {mode}")
        if waiting > 1:
            logger.info(f"📥 요청 들어옴 ({head}){_rid(req_id)} · 큐 {waiting}명 대기 중")
        else:
            logger.info(f"📥 요청 들어옴 ({head}){_rid(req_id)}")
    except Exception as e:
        logger.warning(f"요청 로그 출력 실패: {e}")


def console_log_error(mode: str, user_input: str, err: BaseException, req_id: str | None = None) -> None:
    """생성 처리 중 예외를 콘솔에 남긴다. 파일에는 api 가 status=error 레코드로 따로 남긴다."""
    try:
        head = _MODE_EMOJI.get(mode, f"⚪ {mode}")
        logger.error(f"❌ 생성 실패 ({head}){_rid(req_id)}: {type(err).__name__}: {err}")
        if user_input:
            logger.error(f"   └ 입력: {user_input}")
    except Exception as e:
        logger.warning(f"에러 로그 출력 실패: {e}")


def console_log_rejected(mode: str, ip: str) -> None:
    """동일 IP 중복요청 거절을 콘솔에 남긴다(자동화 연속요청 감지용)."""
    try:
        head = _MODE_EMOJI.get(mode, f"⚪ {mode}")
        logger.warning(f"🚫 중복 IP 거절 ({head}): {ip} — 처리 중인 작업이 있음")
    except Exception as e:
        logger.warning(f"거절 로그 출력 실패: {e}")
