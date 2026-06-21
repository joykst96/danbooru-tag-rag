"""
생성 로그 파일 기록 모듈

사용자 입력 / 최종 태그 / 최종 자연어를 파일에 한 줄(JSON Lines)로 남긴다.
최종 태그는 언더스코어(_)를 공백으로 치환해 저장한다(사용자 요구).

- 경로: 환경변수 GEN_LOG_PATH (기본 /app/logs/generations.jsonl).
  Docker compose 에서 ./logs:/app/logs 볼륨으로 호스트에 보존.
- 단일 워커 전제(api 와 동일). 프로세스 내 단순 append.
- 자정(로컬) 기준 일자별 파일 분리(generations-YYYY-MM-DD.jsonl)로 무한 증식 방지.
- 기록 실패가 응답을 막지 않도록 모든 예외를 삼킨다(로그는 보조 기능).
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# 기본 경로(컨테이너): /app/logs/. 환경변수로 덮어쓰기 가능.
_LOG_PATH = Path(os.environ.get("GEN_LOG_PATH", "/app/logs/generations.jsonl"))


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
) -> None:
    """
    생성 1건을 파일에 기록.

    Args:
        user_input: 사용자가 넣은 원본 입력(기본=한 문장, 분할=구조 요약 문자열).
        final_tags: 최종 태그 리스트(언더스코어 포함된 원본). 저장 시 공백 치환됨.
        nl_prompt: 최종 자연어 프롬프트.
        mode: "basic" | "split".
        extra: 추가로 남길 메타(선택).
    """
    try:
        path = _dated_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "input": user_input,
            "final_tags": _spacify(final_tags),                # 언더스코어→공백
            "final_tags_str": ", ".join(_spacify(final_tags)),  # 사람이 보기 쉬운 한 줄
            "nl_prompt": nl_prompt,
        }
        if extra:
            record["extra"] = extra

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        # 로그 실패는 응답을 막지 않는다.
        logger.warning(f"생성 로그 기록 실패: {e}")


# ── 콘솔(stdout) 로그 ──────────────────────────────────────────────
# 파일 기록(log_generation)과 별개로, 실시간 모니터링용 콘솔 출력.
# docker compose logs -f 로 바로 확인. mode 이모지로 한눈에 구분.

_MODE_EMOJI = {"basic": "🟢 일반 모드", "split": "🔵 고급 모드"}


def console_log_generation(
    user_input: str,
    final_tags: list[str],
    nl_prompt: str,
    mode: str = "basic",
) -> None:
    """생성 1건을 콘솔에 보기 좋게 출력(파일 기록과 독립)."""
    try:
        head = _MODE_EMOJI.get(mode, f"⚪ {mode}")
        tags_str = ", ".join(_spacify(final_tags))
        logger.info(head)
        logger.info(f"📝 사용자 프롬프트: {user_input}")
        logger.info(f"✨ 최종 반환 프롬프트: {tags_str}")
        if nl_prompt:
            logger.info(f"   └ 자연어: {nl_prompt}")
    except Exception as e:
        logger.warning(f"콘솔 로그 출력 실패: {e}")
