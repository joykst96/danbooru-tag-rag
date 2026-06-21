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

# ── 분할입력(고급) 전용: 캐릭터/작품 태그 선택 ──
# 사용자가 입력한 캐릭터명·작품명과, DB 검색으로 얻은 후보군을 함께 주고
# LLM이 가장 부합하는 캐릭터 태그와 작품 태그를 고르게 한다.
# (벡터 점수 top1 단순 채택은 "rover" 같은 동명이인/동일작품 내 조연 구분을 못 함.)
SYSTEM_SELECT_CHARACTER = """You select the correct Danbooru character tag and copyright(series) tag.

You are given:
- The user's intended CHARACTER name (as they typed it, possibly Korean or English).
- The user's intended SERIES name (optional).
- CHARACTER_CANDIDATES: Danbooru character tags retrieved from the DB, each with a score and aliases.
- SERIES_CANDIDATES: Danbooru copyright tags retrieved from the DB, each with a score and aliases.

Pick the single best character tag and the single best series tag from the candidates.

RULES:
1. Choose ONLY from the provided candidates. Never invent a tag not in the lists. If nothing fits, return empty string for that field.
2. The user's intent dominates the vector score. A lower-scored candidate that matches the user's stated name/series is correct over a higher-scored mismatch.
3. If a SERIES name is given, prefer the character candidate that belongs to that series. Danbooru character tags look like "name_(series)" — match the series inside the parentheses to the chosen series tag.
4. Disambiguate within the same series by the character NAME the user typed (e.g. user "프리렌"/"frieren" → pick "frieren", NOT a side character like "linie_(sousou_no_frieren)").
5. Match across languages and aliases (Korean name vs English tag vs alias).
6. If the user's character clearly is not among the candidates, return "" for character (do not force a wrong pick).

Return ONLY a JSON object, no markdown:
{"character": "<tag or empty>", "series": "<tag or empty>"}"""

# ── 분할입력(고급) 전용: 인물 단위 자연어 프롬프트 ──
# Anima 공식 허깅페이스 팁:
#   "Name a character, then describe their basic appearance."
#   다인물일수록 중요 — 이름만 나열하고 외형묘사가 없으면 모델이 인물을 헷갈린다.
# 따라서 각 인물마다 [이름(+작품)] + [외형/속성 묘사]를 반드시 한 덩어리로 붙여 쓴다.
# 캐릭터명이 DB에 없어 태그가 안 잡힌 경우에도 사용자가 적은 이름 문자열을 그대로
# 살려 쓰되, 그 칸의 묘사 태그로 외형을 채운다(이름만 덩그러니 두지 않는다).
SYSTEM_NL_MULTI = """You are an expert prompt engineer for the ANIMA anime image generation model.

You are given a SCENE made of one or more CHARACTER blocks and an optional BACKGROUND block. Write a single descriptive English prompt for the whole scene.

CORE RULE (critical for multi-character scenes):
For EACH character, name the character first, then immediately describe their appearance. Never list character names without appearance description — the model confuses characters otherwise. Even when a name is given, weave in the appearance tags for that character.

WHEN THERE ARE 2 OR MORE CHARACTERS:
Describe each character in its own SEPARATE sentence. Do NOT pack multiple characters into one sentence, and do NOT blend their attributes. One sentence per character, each starting with that character (e.g. "On the left, <Name A> ... ." then "On the right, <Name B> ... ."). Keeping characters in distinct sentences is what prevents the model from mixing them up.

RULES:
1. Write each character as its own clause/sentence: "<Name> from <Series>, with <appearance woven from the tags>, ...". If a character has a series, include it once.
2. If a block has a name but NO matched tags, still use the name and describe whatever the block's description implies — do not drop the character.
3. If a block is marked original_character=true, it has no canon name. Invent a simple fitting first name for that character and use it the same way (name first, then appearance). This gives the model a stable handle so it does not confuse this character with others.
4. If a block has NO name and is not original, use a plain noun ("A girl", "A boy", "A figure") consistent with the description, then the appearance.
5. Keep every character visually distinct so the model can separate them. Do not merge two characters' attributes into the same sentence.
6. End with the background/setting as a shared clause if a BACKGROUND block is given.
7. Use standard capitalization for character/series names. No 'beautiful'/'cute' filler. At least one full sentence per character.
8. Return ONLY the raw prompt text. No markdown, no headings, no explanations."""


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


async def select_character_and_series(
    user_name: str,
    user_series: str,
    char_candidates: list[dict],
    series_candidates: list[dict],
    temperature: float = 0.0,
) -> tuple[str, str]:
    """
    [분할입력 캐릭터/작품 선택] 사용자 입력 + DB 후보군을 LLM에 주고 올바른 태그 선택.

    char_candidates / series_candidates 각 원소: {"tag": str, "score": float, "aliases": list[str]}
    Returns: (character_tag, series_tag)  — 못 고르면 빈 문자열.
    """
    if not user_name and not user_series:
        return "", ""

    def _fmt(cands: list[dict]) -> str:
        lines = []
        for c in cands:
            al = ", ".join(c.get("aliases", [])[:6])
            lines.append(
                f'- {c["tag"]} (score={c.get("score", 0):.3f}'
                + (f', aliases=[{al}]' if al else "")
                + ")"
            )
        return "\n".join(lines) if lines else "(none)"

    user = (
        f'CHARACTER name: "{user_name}"\n'
        f'SERIES name: "{user_series}"\n\n'
        f"CHARACTER_CANDIDATES:\n{_fmt(char_candidates)}\n\n"
        f"SERIES_CANDIDATES:\n{_fmt(series_candidates)}"
    )

    try:
        content = await _call(SYSTEM_SELECT_CHARACTER, user, temperature)
    except LLMError:
        return "", ""

    # JSON 객체 파싱 (마크다운 펜스 제거 후)
    txt = _strip_markdown(content)
    import json, re as _re
    m = _re.search(r"\{.*\}", txt, _re.DOTALL)
    if not m:
        return "", ""
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return "", ""

    char_tag = (obj.get("character") or "").strip()
    series_tag = (obj.get("series") or "").strip()

    # 환각 방지: 후보에 실제 존재하는 태그만 채택
    char_set = {c["tag"] for c in char_candidates}
    series_set = {c["tag"] for c in series_candidates}
    if char_tag not in char_set:
        char_tag = ""
    if series_tag not in series_set:
        series_tag = ""
    return char_tag, series_tag


async def generate_nl_multi(
    characters: list[dict],
    background_tags: list[str] | None = None,
    background_desc: str = "",
    temperature: float = 0.4,
) -> str:
    """
    [분할입력 자연어] 인물 블록 + 배경으로 인물 단위 자연어 프롬프트 생성.

    Args:
        characters: 인물 블록 리스트. 각 원소:
            {
              "name": str,        # 사용자가 적은 캐릭터명(공백 가능)
              "series": str,      # 작품명(공백 가능)
              "tags": list[str],  # 그 인물 칸에서 확정된 태그(캐릭터/작품/속성)
              "desc": str,        # 원본 캐릭터묘사(한국어, 폴백 설명용)
            }
        background_tags: 배경칸에서 확정된 태그
        background_desc: 배경칸 원본(한국어)

    이름만 나열되지 않도록 SYSTEM_NL_MULTI 가 각 인물에 외형묘사를 강제한다.
    실패 시 빈 문자열(자연어는 보조 수단).
    """
    lines: list[str] = []
    for i, ch in enumerate(characters, 1):
        parts = [f"CHARACTER {i}:"]
        if ch.get("is_original"):
            # 오리지널 캐릭터: 작품/캐릭터명 없음. LLM이 임의 이름을 지어 붙인다.
            parts.append("original_character=true (no canon name — invent a fitting name)")
        else:
            if ch.get("name"):
                parts.append(f'name="{ch["name"]}"')
            if ch.get("series"):
                parts.append(f'series="{ch["series"]}"')
        if ch.get("tags"):
            parts.append(f'tags=[{", ".join(ch["tags"])}]')
        if ch.get("desc"):
            parts.append(f'description="{ch["desc"]}"')
        lines.append("  ".join(parts))

    if background_tags or background_desc:
        bparts = ["BACKGROUND:"]
        if background_tags:
            bparts.append(f'tags=[{", ".join(background_tags)}]')
        if background_desc:
            bparts.append(f'description="{background_desc}"')
        lines.append("  ".join(bparts))

    user = "\n".join(lines)
    try:
        content = await _call(SYSTEM_NL_MULTI, user, temperature)
        return _strip_markdown(content)
    except LLMError:
        return ""
