"""
LLM 모듈

로컬 LLM(llama.cpp 등 OpenAI 호환 엔드포인트) 호출 함수들.
pipeline 과 benchmark 양쪽에서 재사용한다.

모든 함수는 thinking off, 낮은 temperature(변별 작업) 전제로 설계되었으나
temperature 는 인자로 조절 가능하다. 자연어 생성만 약간 높은 temperature 권장.
"""

import re
import json
import logging

import httpx

from .config import (
    LLM_API_URL,
    LLM_MODEL,
    LLM_THINKING_ENABLED,
)
from .exceptions import LLMError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 시스템 프롬프트
# ---------------------------------------------------------------------------
SYSTEM_KEYWORDS = """You are an expert anime illustration tag extractor.
Convert the Korean prompt into a JSON array of specific English keywords.
Break down visual elements: character count, hair/eye traits, clothing, actions, objects, background.
Prefer concrete Danbooru-style phrasing (e.g. '1girl', 'blue hair', 'school uniform').
Return ONLY a valid JSON array of strings. No markdown, no extra text.
Example: ["1girl", "blue hair", "school uniform", "rain", "street"]"""

SYSTEM_REFINE = """You are a Danbooru tag matching assistant.
Given a Korean intent and a list of candidate Danbooru tags retrieved from a database,
select and refine the English tag expressions that best match the user's actual intent.
The candidates are real tags from the database; prefer them, but you may adjust wording
to better English tag form if needed.
Return ONLY a valid JSON array of English strings. No markdown, no extra text."""

SYSTEM_SELECT = """You are a strict Danbooru tag selector.
Given a Korean intent and structured candidate tags (with categories),
choose the final set of tags that faithfully represent the intent.

CRITICAL: You may ONLY output tags that appear VERBATIM in the candidate list.
Never invent, complete, normalize, or modify a tag. If the intent is "은발"
(silver hair) and the candidate list contains "grey_hair" but not "silver_hair",
you MUST output "grey_hair" — outputting "silver_hair" is forbidden because it is
not in the candidates. Copy tags exactly as given, character for character.

Exclude tags pulled in only by vector similarity but not actually described.
Return ONLY a valid JSON array of the chosen English tag strings. No markdown."""

SYSTEM_NL = """You are an expert prompt engineer for the ANIMA image generation model.
Given a Korean intent and English keywords, write a descriptive natural-language English prompt.
Rules:
1. At least 2 sentences. More descriptive is better.
2. Standard capitalization for character/series names.
3. If no specific character, use plain nouns like 'A girl', 'A boy'. No 'beautiful'/'cute' embellishment.
4. Return ONLY the raw prompt text. No markdown, no explanations."""

SYSTEM_VERIFY = """You are a strict tag reviewer for Danbooru tags.
Identify tags from the 'Matched Tags' list that are unrelated to the 'Korean Intent'.
Return ONLY a JSON array of suspicious tag strings. Empty array [] if all are fine. No markdown."""


# ---------------------------------------------------------------------------
# 공통 호출 헬퍼
# ---------------------------------------------------------------------------
def _build_payload(system: str, user: str, temperature: float) -> dict:
    """OpenAI 호환 chat completions payload 구성 (thinking 제어 포함)."""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "top_p": 0.9,
    }
    # thinking 비활성 (llama.cpp / vLLM 컨벤션)
    if LLM_THINKING_ENABLED:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    else:
        payload["thinking_budget_tokens"] = 0
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


async def _call(system: str, user: str, temperature: float, timeout: float = 120.0) -> str:
    """LLM 호출 후 content 문자열 반환."""
    payload = _build_payload(system, user, temperature)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(LLM_API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise LLMError(f"LLM 호출 실패: {e}")


def _extract_json_array(content: str) -> list[str]:
    """응답에서 JSON 문자열 배열을 추출. 실패 시 빈 리스트."""
    match = re.search(r'\[.*\]', content, re.DOTALL)
    if not match:
        return []
    try:
        arr = json.loads(match.group(0))
        if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
            return arr
    except json.JSONDecodeError:
        pass
    return []


def _strip_markdown(text: str) -> str:
    """자연어 응답의 마크다운 코드블록/따옴표 제거."""
    text = re.sub(r'^```[\w]*\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
    return text.strip().strip('"')


# ---------------------------------------------------------------------------
# 단계별 함수
# ---------------------------------------------------------------------------
async def extract_keywords(korean_prompt: str, temperature: float = 0.1) -> list[str]:
    """[Pass 1 전처리] 한국어 → 영어 키워드 배열."""
    content = await _call(SYSTEM_KEYWORDS, korean_prompt, temperature)
    kws = _extract_json_array(content)
    if not kws:
        raise LLMError(f"키워드 추출 실패. 응답: {content[:200]}")
    return kws


async def refine_with_candidates(
    korean_prompt: str,
    candidate_tags: list[str],
    temperature: float = 0.1,
) -> list[str]:
    """
    [2-pass 핵심] 후보 태그를 보여주고 영어 의도표현으로 정제.
    '번역→태그'가 아니라 '태그 보고 의도 매핑' (mode collapse 회피).
    """
    user = (
        f"Korean intent: {korean_prompt}\n"
        f"Candidate tags from DB: {', '.join(candidate_tags)}"
    )
    content = await _call(SYSTEM_REFINE, user, temperature)
    return _extract_json_array(content)


async def select_final(
    korean_prompt: str,
    grouped_candidates: dict[str, list[str]],
    temperature: float = 0.1,
) -> list[str]:
    """
    [최종 선택] 카테고리 구조화된 후보 중 의도에 맞는 것 선택.
    grouped_candidates 예: {"general": [...], "character": [...], "copyright": [...]}
    """
    lines = [f"Korean intent: {korean_prompt}", "Candidates by category:"]
    for label, tags in grouped_candidates.items():
        if tags:
            lines.append(f"  [{label}] {', '.join(tags)}")
    content = await _call(SYSTEM_SELECT, "\n".join(lines), temperature)
    return _extract_json_array(content)


async def generate_nl_prompt(
    korean_prompt: str,
    keywords: list[str],
    temperature: float = 0.4,
) -> str:
    """[자연어 생성] Anima용 자연어 영어 프롬프트. temperature 약간 높게."""
    user = f"Korean intent: {korean_prompt}\nKeywords: {', '.join(keywords)}"
    try:
        content = await _call(SYSTEM_NL, user, temperature)
        return _strip_markdown(content)
    except LLMError:
        return ""  # 자연어는 보조 수단이라 실패해도 전체를 막지 않음


async def verify_tags(
    korean_prompt: str,
    tags: list[str],
    temperature: float = 0.1,
) -> list[str]:
    """[검증] 의도와 무관한 의심 태그 식별 (제거가 아니라 표시용)."""
    if not tags:
        return []
    user = f"Korean intent: {korean_prompt}\nMatched tags: {', '.join(tags)}"
    try:
        content = await _call(SYSTEM_VERIFY, user, temperature)
        return _extract_json_array(content)
    except LLMError:
        return []  # 검증 실패는 무시 (보조 기능)
