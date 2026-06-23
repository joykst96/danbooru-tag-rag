"""
분해 파이프라인 (측정용, 4회 호출 구조)

본선 run_pipeline(2-pass) 과 별개. 라이프사이클에서 합의한 4회 구조를
실제로 돌려보며 중간 산출물을 전부 노출한다. 측정 목적이므로 본선에
박지 않는다(추측 박기 방지). test.html 에서 단계별로 확인.

흐름 (LLM 4회):
  1. 한국어 분해 + 질의(일반 cat0)          → ko_units, A(단어–후보군)
  2. 통번역 분해 + 질의(일반 cat0)          → en_units, B(단어–후보군)
     (인원수 단위는 검색 거치되 후보 보존, 캐릭터 단위는 캐릭터DB+폴백)
  3. A∪B 후보풀 + 원본 → 태그 완성          → final_tags (+ DB 실존 코드필터)
  4. 영어 단위 + 원본 → 자연어 프롬프트

각 변형(평탄 vs 쌍유지, 4 vs 6회)은 측정으로 정한다. 현재는
'쌍 구조 유지하되 3번은 평탄 후보풀로 합쳐 원본 기준 선별' 로 고정.
"""

import asyncio
import logging
from dataclasses import dataclass, field

from . import search as search_mod
from . import llm
from . import llm_decompose as dec
from .config import DEFAULT_TOP_K

logger = logging.getLogger(__name__)


@dataclass
class DecomposedResult:
    korean_prompt: str
    ko_units: list[str] = field(default_factory=list)           # 1. 한국어 분해 단위
    en_units: list[str] = field(default_factory=list)           # 2. 영어 분해 단위(번역후)
    person_units: list[str] = field(default_factory=list)       # 인원수 단위
    ko_candidates: dict[str, list[str]] = field(default_factory=dict)  # A
    en_candidates: dict[str, list[str]] = field(default_factory=dict)  # B
    candidate_pool: list[str] = field(default_factory=list)     # A∪B 합본
    final_tags: list[str] = field(default_factory=list)         # 3. 완성(필터후)
    hallucinated: list[str] = field(default_factory=list)       # DB에 없어 제거
    nl_prompt: str = ""                                          # 4. 자연어


# 인원수 단위 식별 — 단어경계 고려(부분문자열 오분류 방지: "둘러싼"≠인원수)
import re
_PERSON_PATTERNS = [
    r"\b\d+\s*(girl|boy|other|people|man|woman|명|인)\b",
    r"(여자|여성|소녀|여캐|남자|남성|소년|남캐|커플|couple)",
    r"(두 명|세 명|네 명|다섯|혼자|솔로|solo|multiple|group)",
    r"(여자|남자|사람)\s*(둘|셋|넷|하나|한 명|두 명)",
]
_PERSON_RE = re.compile("|".join(_PERSON_PATTERNS))


def looks_like_person_unit(unit: str) -> bool:
    """인원수/인물 관련 단위인지. 단어경계 기반(부분문자열 오분류 회피)."""
    return bool(_PERSON_RE.search(unit.lower()))


# ---------------------------------------------------------------------------
# 일반 모드 캐릭터 해석 (2단계: 쌍 추출 → cat3/cat4 검색 → 일괄 판별)
# ---------------------------------------------------------------------------
async def _resolve_characters(
    korean_prompt: str, variant: str, top_k: int,
) -> tuple[list[str], list[str]]:
    """
    한 문장에서 (작품?, 캐릭터) 쌍을 추출·판별해 확정 캐릭터 태그를 얻는다.

    절차:
      1) LLM 이 (작품?, 캐릭터) 쌍 배열 추출. 없으면 ([], []) 즉시 반환(오버헤드 0).
      2) 각 쌍을 cat4(캐릭터)/cat3(작품) top_k 검색.
      3) 모든 쌍의 후보를 한 번에 LLM 에 줘서 쌍별 캐릭터 태그 선택(환각=후보 밖 제거).

    Returns:
      (char_tags, char_names)
        char_tags : 확정된 Danbooru 캐릭터 태그(작품 태그는 미포함 — 최종/NL 모두 제외).
        char_names: 사용자가 적은 캐릭터/작품 이름들(일반 검색 텍스트에서 제거할 대상).
    """
    pairs = await llm.extract_character_pairs(korean_prompt)
    if not pairs:
        return [], []

    def _cands(hits) -> list[dict]:
        out = []
        for r in (getattr(hits, "results", None) or []):
            out.append({
                "tag": r.tag,
                "score": r.score,
                "aliases": getattr(r, "aliases", []) or [],
            })
        return out

    enriched = []
    names: list[str] = []
    for pr in pairs:
        name = pr.get("character", "").strip()
        series = pr.get("series", "").strip()
        if name:
            names.append(name)
        if series:
            names.append(series)
        char_cands: list[dict] = []
        series_cands: list[dict] = []
        if name:
            c_hits = search_mod.search_character_only([name], variant, top_k=top_k)
            if c_hits:
                char_cands = _cands(c_hits[0])
        if series:
            s_hits = search_mod.search_copyright([series], variant, top_k=top_k)
            if s_hits:
                series_cands = _cands(s_hits[0])
        enriched.append({
            "character": name, "series": series,
            "char_candidates": char_cands, "series_candidates": series_cands,
        })

    chosen = await llm.select_character_pairs(enriched)
    char_tags = []
    for t in chosen:
        if t and t not in char_tags:
            char_tags.append(t)
    return char_tags, names


def _strip_units(units: list[str], names: list[str]) -> list[str]:
    """분해 단위 리스트에서 캐릭터/작품 이름이 섞인 단위를 제거·정리.

    이름이 통째로 한 단위면 그 단위를 버리고, 단위 안에 이름이 일부 포함되면
    그 부분만 지운다(남은 텍스트가 비면 단위 제거). 캐릭터 정체성은 cat4 확정태그가
    담당하므로 일반 검색은 동작/속성만 찾으면 된다.
    """
    out = []
    for u in units:
        stripped = search_mod.strip_names_from_text(u, names).strip()
        if stripped:
            out.append(stripped)
    return out


async def run_decomposed_pipeline(
    korean_prompt: str,
    general_variant: str = "c",
    en_variant: str = "b",
    top_k: int = 5,
    generate_nl: bool = True,
    search_categories: set[int] | None = None,
    nl_tone: str | None = None,
) -> DecomposedResult:
    """4회 분해 파이프라인 실행. 중간 산출물을 전부 담아 반환."""
    res = DecomposedResult(korean_prompt=korean_prompt)

    # ── 호출 1: 한국어 분해 + 질의 ──
    res.ko_units = await dec.decompose_korean(korean_prompt)
    ko_attr = [u for u in res.ko_units if not looks_like_person_unit(u)]
    res.person_units = [u for u in res.ko_units if looks_like_person_unit(u)]
    if ko_attr:
        ko_hits = search_mod.search_pool(ko_attr, general_variant, top_k=top_k, categories=search_categories)
        res.ko_candidates = {h.keyword: [r.tag for r in h.results] for h in ko_hits}

    # ── 호출 2: 통번역 분해 + 질의 ──
    res.en_units = await dec.translate_decompose(korean_prompt)
    if res.en_units:
        # 인원수 영어단위(2girls 등)도 일반DB 질의(실존 태그 확정 위해)
        en_hits = search_mod.search_pool(res.en_units, en_variant, top_k=top_k, categories=search_categories)
        res.en_candidates = {h.keyword: [r.tag for r in h.results] for h in en_hits}

    # ── 후보풀 합본 (A ∪ B) ──
    pool: list[str] = []
    for cands in res.ko_candidates.values():
        pool.extend(cands)
    for cands in res.en_candidates.values():
        pool.extend(cands)
    # dedup 순서유지
    seen, dedup = set(), []
    for t in pool:
        if t not in seen:
            seen.add(t); dedup.append(t)
    res.candidate_pool = dedup

    # ── 호출 3: 태그 완성 (원본 기준) + 환각 코드필터 ──
    raw_final = await dec.complete_tags(korean_prompt, res.candidate_pool)
    # 환각 차단: 후보풀에도 없고 DB에도 없으면 제거 (이중 안전)
    pool_set = set(res.candidate_pool)
    in_pool = [t for t in raw_final if t in pool_set]
    out_pool = [t for t in raw_final if t not in pool_set]
    # 풀에 있어도 DB 실존 재확인(번역분해가 가짜태그 섞을 가능성 차단)
    kept, dropped_db = search_mod.filter_existing_tags(in_pool, en_variant)
    res.final_tags = kept
    res.hallucinated = out_pool + dropped_db
    if res.hallucinated:
        logger.warning(f"환각 제거(풀밖 {out_pool} / DB밖 {dropped_db})")

    # ── 호출 4: 자연어 프롬프트 (최종 태그 + 원본) ──
    # 환각 필터를 거친 final_tags 기준으로 생성(없는 태그가 NL에 새지 않게).
    if generate_nl:
        nl_basis = res.final_tags or res.en_units
        res.nl_prompt = await llm.generate_nl_prompt(korean_prompt, nl_basis, tone=nl_tone)

    return res


async def stream_decomposed_pipeline(
    korean_prompt: str,
    general_variant: str = "b",
    en_variant: str = "b",
    top_k: int = 5,
    generate_nl: bool = True,
    search_categories: set[int] | None = None,
    nl_tone: str | None = None,
):
    """
    4회 분해 파이프라인을 단계별로 스트리밍 (async generator).

    각 단계가 끝날 때마다 (stage, status, data) 를 yield 한다.
    api 가 이를 SSE 로 흘려 UI 가 키워드/최종을 도착 즉시 표시하게 한다.
    동시 요청이 겹쳐도 사용자가 진행을 보며 기다리도록(새로고침 감소).

    yield 형식: {"stage": str, "status": str, "data": dict}
      - "korean"   : 한국어 분해 단위 (ko_units, person_units)
      - "english"  : 번역+분해 영어 단위 (en_units)
      - "final"    : 최종 태그 + 환각 (final_tags, hallucinated)
      - "nl"       : 자연어 프롬프트 (nl_prompt)
    """
    # ── 0단계: 캐릭터 쌍 추출 + 판별 (일반 모드 캐릭터 지원) ──
    # 한 문장에서 (작품?, 캐릭터) 쌍을 LLM이 뽑아 cat3/cat4 검색 후 일괄 판별.
    # 없으면 빈 결과 → 기존 흐름 그대로(오버헤드 최소). 확정된 캐릭터 태그는 최종에
    # 합본하고, 추출된 이름은 일반 검색 텍스트에서 제거해 _(cosplay) 오염을 막는다.
    char_tags, char_names = await _resolve_characters(korean_prompt, general_variant, top_k)

    # ── 1단계: 한국어 분해 (+ 질의) ──
    ko_units = await dec.decompose_korean(korean_prompt)
    ko_attr = [u for u in ko_units if not looks_like_person_unit(u)]
    person_units = [u for u in ko_units if looks_like_person_unit(u)]
    yield {
        "stage": "korean",
        "status": "한국어 키워드 추출 완료",
        "data": {"ko_units": ko_units, "person_units": person_units},
    }

    # 추출된 캐릭터 이름을 분해 단위에서 제거(일반검색 오염 차단).
    if char_names:
        ko_attr = _strip_units(ko_attr, char_names)

    ko_candidates: dict[str, list[str]] = {}
    if ko_attr:
        ko_hits = search_mod.search_pool(ko_attr, general_variant, top_k=top_k, categories=search_categories)
        ko_candidates = {h.keyword: [r.tag for r in h.results] for h in ko_hits}

    # ── 2단계: 통번역 분해 (+ 질의) ──
    en_units = await dec.translate_decompose(korean_prompt)
    yield {
        "stage": "english",
        "status": "영어 키워드 추출 완료",
        "data": {"en_units": en_units},
    }

    if char_names:
        en_units = _strip_units(en_units, char_names)

    en_candidates: dict[str, list[str]] = {}
    if en_units:
        en_hits = search_mod.search_pool(en_units, en_variant, top_k=top_k, categories=search_categories)
        en_candidates = {h.keyword: [r.tag for r in h.results] for h in en_hits}

    # ── 후보풀 합본 ──
    pool: list[str] = []
    for cands in ko_candidates.values():
        pool.extend(cands)
    for cands in en_candidates.values():
        pool.extend(cands)
    seen, candidate_pool = set(), []
    for t in pool:
        if t not in seen:
            seen.add(t); candidate_pool.append(t)

    # ── 3단계: 태그 완성 + 환각 코드필터 ──
    raw_final = await dec.complete_tags(korean_prompt, candidate_pool)
    pool_set = set(candidate_pool)
    in_pool = [t for t in raw_final if t in pool_set]
    out_pool = [t for t in raw_final if t not in pool_set]
    kept, dropped_db = search_mod.filter_existing_tags(in_pool, en_variant)
    hallucinated = out_pool + dropped_db
    if hallucinated:
        logger.warning(f"환각 제거(풀밖 {out_pool} / DB밖 {dropped_db})")
    if not kept:
        logger.warning(f"⚠️ 최종 태그 빔 — raw_final={raw_final}, in_pool={in_pool}, dropped_db={dropped_db}")
    # 확정 캐릭터 태그를 맨 앞에 합본(작품 태그는 미포함). 묘사검색이 _(cosplay) 등을
    # 끌어왔으면 후처리로 제거(캐릭터 정체성은 cat4 확정태그가 담당).
    kept = search_mod.drop_character_derived(kept)
    if char_tags:
        kept = char_tags + [t for t in kept if t not in char_tags]
    yield {
        "stage": "final",
        "status": "태그 완성",
        "data": {"final_tags": kept, "hallucinated": hallucinated},
    }

    # ── 4단계: 자연어 프롬프트 ──
    nl_prompt = ""
    if generate_nl:
        nl_basis = kept or en_units
        nl_prompt = await llm.generate_nl_prompt(korean_prompt, nl_basis, tone=nl_tone)
    yield {
        "stage": "nl",
        "status": "완료",
        "data": {"nl_prompt": nl_prompt},
    }
