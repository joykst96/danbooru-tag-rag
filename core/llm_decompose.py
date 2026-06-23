"""
한국어 의미단위 분해 + 분해기반 선별 모듈 (측정용)

기존 extract_keywords 는 한국어 → 영어 키워드로 한 번에 변환한다. 그 과정에서
LLM이 세부표현을 head 단어로 뭉개는(mode collapse) 수렴이 일어난다고 관측됨.

이 모듈은 다른 경로를 제공한다:
    1. decompose_korean : 한국어 문장을 '검색 단위' 한국어 조각으로 쪼갬 (영어 변환 X)
       → 각 조각을 e5 cross-lingual 로 직접 질의 (수렴 우회 시도)
    2. select_from_decomposed : 조각별로 끌어온 후보(영어 태그)를 받아 의도에 맞게 선별
       → 인원수는 검색에 맡기지 않고 분해 단계의 인원 단위에서 직접 확정 (붕괴 방지)

llm.py 의 _call / _extract_json_array / _build_payload 를 재사용한다.
benchmark 에서 기존 경로와 나란히 비교하기 위한 것이며, 효과 확인 전까지는
파이프라인 본선에 넣지 않는다 (추측으로 박지 말 것).
"""

import json
import logging

from . import llm  # _call, _extract_json_array 재사용

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1) 분해 프롬프트
# ---------------------------------------------------------------------------
# 설계 원칙:
#   - 의미단위 = 하나의 시각적 속성 (태그 1개에 대응하도록)
#   - 쪼개면 의미가 깨지는 건 한 덩어리로 유지 ("머리 양쪽 드래곤 뿔")
#   - 인원수/관계는 반드시 별도 단위로 분리 (인원수 붕괴 방지의 출발점)
#   - 한국어를 그대로 유지 (영어 변환 금지 — cross-lingual 질의에 맡김)
#   - 입력에 섞인 영어 표현은 그대로 한 단위로 보존 (사용자 코드스위칭 존중)
SYSTEM_DECOMPOSE = """You split a Korean image-description sentence into search units.

A "search unit" is one visual attribute that maps to roughly one tag: a hair color, a hairstyle, an eye trait, a clothing item, a pose, an action, an object, a background element, an expression, a body type, an age impression, etc.

RULES:
1. Keep each unit in its ORIGINAL language. Do NOT translate Korean to English. If the input already contains an English phrase (code-switching), keep that English phrase as its own unit, unchanged.
2. Split compound descriptions into separate units, BUT keep a phrase together if splitting would lose its meaning. Example: "머리 양쪽 드래곤 뿔" stays as one unit (splitting "양쪽" off makes it meaningless).
3. Person count and relations are ALWAYS their own units. "여자 둘이 남자 하나를 둘러싼" → ["여자 둘", "남자 하나", "둘러싼"]. Never merge counts into other attributes.
4. Drop pure filler that carries no visual meaning (e.g. "~하는 장면", "~인 모습", connective particles alone). Keep anything that affects the image.
5. Preserve named characters or series as single units, as written.

Return ONLY a JSON array of Korean (or original-language) strings. No markdown, no translation, no extra text.
Example input: "은발에 트윈테일 로리, 적안, 머리 양쪽으로 드래곤 뿔, 부끄러워하는 표정"
Example output: ["은발", "트윈테일", "로리", "적안", "머리 양쪽 드래곤 뿔", "부끄러워하는 표정"]"""


# ---------------------------------------------------------------------------
# 1b) 통번역 + 영어 분해 프롬프트
# ---------------------------------------------------------------------------
# 원본 한국어 문장을 통째로 영어로 번역하면서 검색 단위로 분해한다.
# 통 문맥에서 번역해야 "허리까지 오는 구도"→waist_up 처럼 의도가 산다(단위만
# 떼면 의미 깨짐). 구버전도 이 방식. extract_keywords 와 유사하나, 여기서는
# '검색 단위' 관점으로 쪼개는 것을 명시한다.
SYSTEM_TRANSLATE_DECOMPOSE = """You translate a Korean image description into English search units.

Read the WHOLE Korean sentence for context, then output English search units — each a short English tag-like phrase for one visual attribute (hair, eyes, clothing, pose, action, object, background, expression, body, count, etc).

RULES:
1. Translate using the full sentence context, not word-by-word. "허리까지 오는 구도" → "waist up" (framing), not "to the waist".
2. One unit = one attribute. Split compounds, but keep a phrase together if splitting loses meaning ("dragon horns on both sides of head").
3. Person count and relations are their own units ("2girls", "1boy", "surrounded").
4. Keep named characters/series as single units in their common English form.
5. Drop pure filler with no visual meaning.

Return ONLY a JSON array of English strings. No markdown, no Korean, no extra text."""


# ---------------------------------------------------------------------------
# 2) 분해기반 선별 프롬프트
# ---------------------------------------------------------------------------
# 각 한국어 조각마다 DB가 끌어온 후보(영어 태그) 목록을 보여주고,
# 그 조각의 의도에 가장 맞는 태그를 고르게 한다.
#   - 후보는 전부 실존 태그 (환각 불가, 기존 RAG 원칙 유지)
#   - 조각 하나당 0~여러 개 선택 가능 (정확히 맞는 게 없으면 빈 선택)
#   - 인원수 단위는 이 단계 이전에 코드에서 직접 확정하므로 여기 안 옴
SYSTEM_SELECT_DECOMPOSED = """You map Korean search units to real database tags.

You are given several Korean search units. For EACH unit, you receive a list of candidate English tags that were retrieved from a vector database for that unit. The candidates are all real tags that exist in the database.

For each unit, choose the English tag(s) that faithfully express that unit's meaning:
- Pick the candidate(s) that genuinely match the Korean unit. Usually 1, sometimes more, sometimes none.
- If NONE of the candidates actually match the unit's meaning, return nothing for that unit. Do not force a wrong tag just because it was retrieved (the vector search pulls in near-misses).
- Never invent a tag that is not in that unit's candidate list.
- Prefer the more specific tag when it correctly matches (e.g. prefer "aqua_hair" over generic "blue_hair" if the unit means aqua/teal).

Return ONLY a flat JSON array of the chosen English tag strings across all units, deduplicated. No markdown, no per-unit grouping, no extra text."""


# ---------------------------------------------------------------------------
# 2b) 태그 완성 프롬프트 (4회 구조의 3번 — 한/영 후보 합쳐 원본 기준 선별)
# ---------------------------------------------------------------------------
# A(한국어분해 후보) + B(영어분해 후보)를 합친 후보 풀과 원본 한국어 문장을 받아,
# 원본 의도에 맞는 최종 태그를 고른다. 단위정렬은 불필요 — 원본 문장이 기준점.
SYSTEM_COMPLETE = """You assemble the final Danbooru tag set from retrieved candidates.

You receive:
- The original Korean prompt (the source of truth for intent).
- A pool of candidate English tags retrieved from a database (from both Korean-unit and English-unit queries). All candidates are real tags that exist in the database.

Choose the tags that faithfully represent the original prompt's intent.

CRITICAL RULES:
1. You may ONLY output tags that appear VERBATIM in the candidate pool. Never invent, complete, or normalize a tag. If intent is "은발" and the pool has "grey_hair" but not "silver_hair", output "grey_hair". Outputting a tag not in the pool is forbidden.
2. The pool mixes results from two query paths; the same intent may appear as different candidates (e.g. grey_hair vs a near-miss). Pick the one that best matches the original Korean intent.
3. Person count: if the prompt implies multiple people, keep the count tags from the pool (2girls, 1boy, etc). Never collapse a multi-person scene to solo/1girl. Keep exactly the counts the prompt describes.
4. Drop candidates pulled in by similarity but not actually described in the prompt.
5. ORDER: Put person-count tags FIRST, at the very front of the array, before any other tags. These are tags like "1girl", "2girls", "1boy", "multiple_girls", "multiple_boys", "solo", "6+girls", etc. The candidates are mixed in arbitrary order, but the final output MUST lead with the count tags. Example: chosen tags ["grey_hair", "1girl", "smile"] must be output as ["1girl", "grey_hair", "smile"].

Return ONLY a flat JSON array of chosen English tag strings, deduplicated. Output the array and nothing else — no markdown, no preamble, and NO self-review or second-guessing after the array (do not write things like "Wait, let me re-check..."). Decide before you output; the array is your final answer."""


# ---------------------------------------------------------------------------
# 인원수 단위 식별 (분해 결과에서 인원 단위를 코드로 직접 처리)
# ---------------------------------------------------------------------------
# 검색/LLM에 맡기면 58.7% 붕괴하므로, 인원 표현은 분해 조각에서 직접 잡아낸다.
# 키워드 매칭 기반(가벼움). 필요시 확장.
_PERSON_HINTS = (
    "여자", "여성", "소녀", "여캐", "걸", "girl",
    "남자", "남성", "소년", "남캐", "보이", "boy", "man",
    "커플", "couple", "둘", "두 명", "두명", "세 명", "셋",
    "다수", "여러", "multiple", "group", "1인", "혼자", "솔로", "solo",
)


def looks_like_person_unit(unit: str) -> bool:
    """분해 조각이 인원수/인물 관련 단위인지 가벼운 휴리스틱으로 판단."""
    u = unit.lower()
    return any(h in u for h in _PERSON_HINTS)


# ---------------------------------------------------------------------------
# 함수
# ---------------------------------------------------------------------------
async def decompose_korean(korean_prompt: str, temperature: float = 0.1) -> list[str]:
    """한국어 문장을 검색 단위(한국어 유지) 리스트로 분해."""
    content = await llm._call(SYSTEM_DECOMPOSE, korean_prompt, temperature)
    units = llm._extract_json_array(content)
    if not units:
        # 분해 실패 시 통문장 1개로 폴백 (호출측에서 통문장 질의하게)
        logger.warning(f"분해 실패, 통문장 폴백. 응답: {content[:160]}")
        return [korean_prompt]
    return units


async def select_from_decomposed(
    unit_candidates: dict[str, list[str]],
    temperature: float = 0.1,
) -> list[str]:
    """
    조각별 후보(영어 태그)를 받아 의도에 맞는 태그를 선별.

    Args:
        unit_candidates: {한국어조각: [후보태그, ...]} (인원수 조각은 제외하고 넘길 것)
    Returns:
        선별된 영어 태그 리스트 (dedup)
    """
    if not unit_candidates:
        return []

    lines = ["Search units and their candidate tags:"]
    for unit, cands in unit_candidates.items():
        if cands:
            lines.append(f'- "{unit}": {", ".join(cands)}')
        else:
            lines.append(f'- "{unit}": (no candidates)')

    content = await llm._call(SYSTEM_SELECT_DECOMPOSED, "\n".join(lines), temperature)
    return llm._extract_json_array(content)


async def translate_decompose(korean_prompt: str, temperature: float = 0.1) -> list[str]:
    """원본 한국어 문장을 통째로 번역하며 영어 검색 단위로 분해."""
    content = await llm._call(SYSTEM_TRANSLATE_DECOMPOSE, korean_prompt, temperature)
    units = llm._extract_json_array(content)
    if not units:
        logger.warning(f"번역분해 실패. 응답: {content[:160]}")
        return []
    return units


async def complete_tags(
    korean_prompt: str,
    candidate_pool: list[str],
    temperature: float = 0.1,
) -> list[str]:
    """
    한/영 후보를 합친 풀 + 원본 한국어로 최종 태그 완성 (4회 구조의 3번).

    환각 금지·인원수 보호는 프롬프트(SYSTEM_COMPLETE)로 1차 강제하되,
    호출측(파이프라인)에서 DB 실존 대조 코드필터를 반드시 한 번 더 건다.
    """
    if not candidate_pool:
        return []
    # 중복 제거(순서 유지)
    seen, pool = set(), []
    for t in candidate_pool:
        if t not in seen:
            seen.add(t)
            pool.append(t)

    user = (
        f"Original Korean prompt:\n{korean_prompt}\n\n"
        f"Candidate tag pool ({len(pool)} tags):\n{', '.join(pool)}"
    )
    content = await llm._call(SYSTEM_COMPLETE, user, temperature)
    parsed = llm._extract_json_array(content)
    if not parsed:
        import logging
        logging.getLogger("core.llm_decompose").warning(
            f"⚠️ complete_tags 파싱 실패 — LLM 원문(앞 500자): {content[:500]!r}"
        )
    return parsed
