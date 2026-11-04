"""
v3 파이프라인 (기본 모드 본선, config.PIPELINE="v3")

tag-embedding 벤치(gen_test/gemma_v3.py)에서 gemma4 26B A4B 로 최적화·측정한 흐름을 그대로 옮겼다.
벤치 10문장 개념 충족률: 기존 4회 분해(v1) 82% → v3 95% (같은 조건 Claude 95~98%), 두 번 실행 10/10 동일.

흐름 (LLM 호출: 분해 1 + 선택 1 + 자연어 1):
  1. 구조화 분해 (JSON 스키마 강제)
       people     : 인원 → 코드가 1girl/2girls/1boy... 로 변환 (인원수를 검색·LLM 선별에 맡기지 않음)
       characters : 고유 캐릭터·작품 이름 {ko, en}
       units      : 나머지 시각 개념, 조각 하나에 개념 하나 {ko, en}
     기존 v1 의 실패: "단발머리 소년" 처럼 인물에 속성이 붙은 조각이 인원수 정규식에 걸려 통째로
     검색에서 빠짐(short_hair, glasses, miko, spacesuit 손실) → 인물·속성 분리 규칙으로 해결.
  2. 조각별 후보 = 어휘 일치(영어 표기와 같은 이름의 태그) + 한국어 질의 top5 + 영어 질의 top5
     (카테고리는 where 사전 필터. 일반 조각 cat0(또는 사용자가 고른 검색 풀), 캐릭터 조각 cat3/4)
  3. 선택: 후보 "번호"만 고를 수 있는 JSON 스키마 → 후보 밖 태그는 구조적으로 불가.
     맞는 후보가 없으면 짧은 영어 구, 그것도 없으면 분해 때 받은 영어(en)를 구로 쓴다(조각을 버리지 않음).
  4. 최종 = 인원수 태그 + 캐릭터 태그(cat4) + 조각별 선택(태그·구).
     작품 태그(cat3)는 기존 정책대로 최종에서 뺀다. DB 실존 확인·_(cosplay) 제거는 기존과 같은 이중 방어.
  5. 자연어: 기존 톤 기능(rich/plain/단어형+)을 그대로 쓴다 (pipeline_decomposed._generate_nl_or_phrase).

스트리밍 이벤트(stage: korean / english / final / nl)는 기존 stream_decomposed_pipeline 과 같은 모양이라
api.py·index.html 은 그대로 동작한다. final_tags 에는 태그 외에 영어 구가 섞일 수 있다(쉼표 제거).
"""

import logging

from . import search as search_mod
from .config import MAIN_VARIANT, V3_SEED
from .exceptions import LLMError
from .llm import _call_json
from .llm_v3 import SYSTEM_DECOMPOSE, SCHEMA_DECOMPOSE, SYSTEM_SELECT
from .pipeline_decomposed import _generate_nl_or_phrase

logger = logging.getLogger(__name__)

TOP_K = 5                 # 질의(한국어·영어)마다 top5 → 조각당 후보 최대 10 + 어휘 일치
DEF_CHARS = 90            # 선택 프롬프트에 보여 줄 정의 길이
ALIAS_COUNT = 4           # 선택 프롬프트에 보여 줄 별칭 수


def count_tags(people: list[dict]) -> list[str]:
    """분해의 people → Danbooru 인원수 태그 (1girl / 2girls ... 6+girls, boy·other 동일)."""
    n = {"girl": 0, "boy": 0, "other": 0}
    for p in people:
        if p.get("type") in n:
            n[p["type"]] += int(p.get("count", 0))
    out = []
    for t, k in (("girl", "girls"), ("boy", "boys"), ("other", "others")):
        c = n[t]
        if c == 1:
            out.append(f"1{t}")
        elif 2 <= c <= 5:
            out.append(f"{c}{k}")
        elif c >= 6:
            out.append(f"6+{k}")
    return out


def candidates(unit: dict, variant: str, cats: set[int]) -> list[dict]:
    """조각 하나의 후보: 어휘 일치 → 한국어 질의 → 영어 질의 순서로 합치고 중복 제거."""
    ko_hits, en_hits = search_mod.search_prefiltered([unit["ko"], unit["en"]], variant, cats, top_k=TOP_K)
    seen, merged = set(), []
    for h in search_mod.lexical_lookup(unit["en"], variant, cats) + ko_hits + en_hits:
        if h["tag"] not in seen:
            seen.add(h["tag"])
            merged.append(h)
    return merged


async def select(korean_prompt: str, units: list[dict], cands: list[list[dict]], name: str) -> tuple[list[list[str]], list[str]]:
    """
    조각별 선택. 반환: (조각별 선택 목록, 메모). 선택은 태그 또는 "phrase:<영어 구>".
    LLM 호출이 실패해도 조각을 버리지 않고 분해 때 받은 영어로 대체한다(메모에 남김).
    """
    lines = [f"Original Korean sentence: {korean_prompt}", ""]
    props = {}
    for i, (u, cs) in enumerate(zip(units, cands)):
        lines.append(f'u{i}: "{u["ko"]}" / {u["en"]}')
        if not cs:
            lines.append("   (no candidates)")
        for j, h in enumerate(cs):
            al = ", ".join(h["aliases"][:ALIAS_COUNT])
            dfn = (h["definition"] or "")[:DEF_CHARS]
            lines.append(f'   {j}. {h["tag"]} — {dfn or "(no definition)"}' + (f" [aliases: {al}]" if al else ""))
        props[f"u{i}"] = {"type": "object", "properties": {
            "pick": {"type": "array", "items": {"type": "integer", "enum": list(range(len(cs))) or [0]}, "maxItems": 3},
            "phrase": {"type": ["string", "null"]}}, "required": ["pick", "phrase"], "additionalProperties": False}
    schema = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

    notes: list[str] = []
    try:
        res = await _call_json(SYSTEM_SELECT, "\n".join(lines), schema, name, seed=V3_SEED)
    except LLMError as e:
        logger.warning(f"v3 선택 실패 → 분해 영어로 대체: {e}")
        notes.append(f"선택 실패: {e}")
        res = {}

    picks: list[list[str]] = []
    for i, (u, cs) in enumerate(zip(units, cands)):
        r = res.get(f"u{i}") or {"pick": [], "phrase": None}
        tags = list(dict.fromkeys(cs[j]["tag"] for j in r.get("pick", []) if 0 <= j < len(cs)))
        if tags:
            picks.append(tags)
        elif r.get("phrase"):
            picks.append(["phrase:" + r["phrase"].strip()])
        else:
            picks.append(["phrase:" + u["en"]])
            notes.append(f"{u['ko']}: 선택·구 없음 → 분해 영어 '{u['en']}'")
    return picks, notes


def _phrase_text(p: str) -> str:
    """구 → 최종 프롬프트 토큰. UI 가 ', ' 로 태그를 나누므로 쉼표는 지운다."""
    return " ".join(p[len("phrase:"):].replace(",", " ").split())


async def stream_v3_pipeline(
    korean_prompt: str,
    variant: str = MAIN_VARIANT,
    generate_nl: bool = True,
    search_categories: set[int] | None = None,
    nl_tone: str | None = None,
):
    """v3 를 단계별로 스트리밍. yield 형식은 stream_decomposed_pipeline 과 같다."""
    unit_cats = set(search_categories) if search_categories else set(search_mod.GENERAL_CATS)

    # ── 1. 구조화 분해 ──
    dec = await _call_json(SYSTEM_DECOMPOSE, korean_prompt, SCHEMA_DECOMPOSE, "decompose", seed=V3_SEED)
    people = dec.get("people", [])
    chars = [c for c in dec.get("characters", []) if c.get("ko")]
    units = [u for u in dec.get("units", []) if u.get("ko")]
    count = count_tags(people)
    logger.info(f"v3 분해: 인원={count} 캐릭터={[c['ko'] for c in chars]} 조각={[u['ko'] + '/' + u['en'] for u in units]}")
    yield {"stage": "korean", "status": "한국어 키워드 추출 완료",
           "data": {"ko_units": [c["ko"] for c in chars] + [u["ko"] for u in units], "person_units": count}}
    en_units = [c["en"] for c in chars] + [u["en"] for u in units]
    yield {"stage": "english", "status": "영어 키워드 추출 완료", "data": {"en_units": en_units}}

    # ── 2. 조각별 후보 ──
    all_units = chars + units
    all_cands = [candidates(c, variant, set(search_mod.CHARACTER_CATS)) for c in chars] + \
                [candidates(u, variant, unit_cats) for u in units]

    # ── 3. 선택 ──
    picks, notes = await select(korean_prompt, all_units, all_cands, variant)
    if notes:
        logger.info(f"v3 선택 메모: {notes}")

    # ── 4. 최종 조립 + 이중 방어 ──
    tag_cat = {h["tag"]: h["category"] for cs in all_cands for h in cs}
    char_tags, body = [], []
    for i, ps in enumerate(picks):
        for p in ps:
            if p.startswith("phrase:"):
                t = _phrase_text(p)
                if t:
                    body.append(t)
            elif i < len(chars):
                if tag_cat.get(p) == 4 and p not in char_tags:   # 작품(cat3)은 최종에서 제외(기존 정책)
                    char_tags.append(p)
            else:
                body.append(p)
    db_tags = [t for t in body if t in tag_cat]
    kept_db, dropped_db = search_mod.filter_existing_tags(db_tags, variant)
    dropped = set(dropped_db)
    body = [t for t in body if t not in dropped]
    body = search_mod.drop_character_derived(body)
    final = list(dict.fromkeys(count + char_tags + body))
    yield {"stage": "final", "status": "태그 완성", "data": {"final_tags": final, "hallucinated": list(dropped_db)}}

    # ── 5. 자연어 (기존 톤 기능 그대로) ──
    nl_prompt = ""
    if generate_nl:
        nl_prompt = await _generate_nl_or_phrase(korean_prompt, final, en_units, nl_tone, variant, char_tags=char_tags)
    yield {"stage": "nl", "status": "완료", "data": {"nl_prompt": nl_prompt}}


async def run_v3_pipeline(korean_prompt: str, variant: str = MAIN_VARIANT, generate_nl: bool = False,
                          search_categories: set[int] | None = None, nl_tone: str | None = None) -> dict:
    """스트림을 모아 dict 로 반환 (검증·벤치마크용)."""
    out: dict = {}
    async for ev in stream_v3_pipeline(korean_prompt, variant, generate_nl, search_categories, nl_tone):
        out.update(ev["data"])
    return out
