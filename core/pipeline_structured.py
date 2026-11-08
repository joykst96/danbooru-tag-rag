"""
structured 파이프라인 (기본 모드 본선. 고급 모드의 묘사·배경 칸도 describe() 로 같은 흐름을 쓴다)
(벤치 실험 이름: v3 → 속도판 v4. 이전 커밋에서는 파일명이 pipeline_v3.py 였다.)

tag-embedding 벤치(gen_test/gemma_v3.py → 속도판 gemma_v4.py)에서 gemma4 26B A4B 로 최적화·측정한 흐름을 옮겼다.
벤치 10문장 개념 충족률: 기존 4회 분해(v1) 82% → v3 95% → 속도판 94~95% (같은 조건 Claude 95~98%).
속도판(2026-11-05): 분해+검색+선택 약 30초 → 약 13초. 프롬프트 캐시 on, 출력 JSON 압축, 질의 임베딩 묶음.
  같은 서버 실측: 선택 출력 225 → 34토큰, 분해 출력 160 → 약 80토큰, 시스템 프롬프트 재처리(3~5초) 제거.

흐름 (LLM 호출: 분해 1 + 선택 1 + 자연어 1):
  1. 구조화 분해 (JSON 스키마 강제)
       girls/boys/others : 인원 → 코드가 1girl/2girls/1boy... 로 변환 (인원수를 검색·LLM 선별에 맡기지 않음)
                           동물은 사람 아님, 이름·관계어로 성별 추론, others 는 성별을 알 수 없을 때만
       characters        : 고유 캐릭터·작품 이름 [한국어, 영어]
       units             : 나머지 시각 개념, 조각 하나에 개념 하나 [한국어, 영어]
     기존 v1 의 실패: "단발머리 소년" 처럼 인물에 속성이 붙은 조각이 인원수 정규식에 걸려 통째로
     검색에서 빠짐(short_hair, glasses, miko, spacesuit 손실) → 인물·속성 분리 규칙으로 해결.
  2. 조각별 후보 = 어휘 일치(영어 표기와 같은 이름의 태그) + 한국어 질의 top5 + 영어 질의 top5
     (카테고리는 where 사전 필터. 일반 조각 cat0(또는 사용자가 고른 검색 풀), 캐릭터 조각 cat3/4)
  3. 선택: 조각마다 후보 "번호" 배열 또는 영어 구 문자열만 허용하는 JSON 스키마 → 후보 밖 태그는 구조적으로 불가.
     맞는 후보가 없으면 짧은 영어 구, 그것도 없으면 분해 때 받은 영어(en)를 구로 쓴다(조각을 버리지 않음).
  4. 최종 = 인원수 태그 + 캐릭터 태그(cat4) + 조각별 선택(태그·구).
     작품 태그(cat3)는 기존 정책대로 최종에서 뺀다. DB 실존 확인·_(cosplay) 제거는 기존과 같은 이중 방어.
  5. 자연어: 톤 기능(rich/plain/단어형+) (generate_nl_or_phrase).

스트리밍 이벤트(stage: korean / english / final / nl)는 이전 4회 분해 파이프라인(삭제됨)과 같은 모양이라
api.py·index.html 은 그대로 동작한다. final_tags 에는 태그 외에 영어 구가 섞일 수 있다(쉼표 제거).
"""

import logging

from . import llm
from . import search as search_mod
from .config import MAIN_VARIANT, LLM_SEED
from .exceptions import LLMError
from .llm import _call_json
from .llm_structured import SYSTEM_DECOMPOSE, SCHEMA_DECOMPOSE, SYSTEM_SELECT, TEXT_PATTERN

logger = logging.getLogger(__name__)

TOP_K = 5                 # 질의(한국어·영어)마다 top5 → 조각당 후보 최대 10 + 어휘 일치
DEF_CHARS = 90            # 선택 프롬프트에 보여 줄 정의 길이
ALIAS_COUNT = 4           # 선택 프롬프트에 보여 줄 별칭 수


def count_tags(dec: dict) -> list[str]:
    """분해의 girls/boys/others → Danbooru 인원수 태그 (1girl / 2girls ... 6+girls, boy·other 동일)."""
    out = []
    for key, sing, plur in (("girls", "girl", "girls"), ("boys", "boy", "boys"), ("others", "other", "others")):
        c = int(dec.get(key, 0) or 0)
        if c == 1:
            out.append(f"1{sing}")
        elif 2 <= c <= 5:
            out.append(f"{c}{plur}")
        elif c >= 6:
            out.append(f"6+{plur}")
    return out


_BROKEN = set('[]"')
# 구(phrase) 문자열: 분해와 같은 문자 제한 + 빈 문자열 금지(패턴이 있으면 llama.cpp 는 minLength 를 보지 않는다)
PHRASE_PATTERN = TEXT_PATTERN.replace("*$", "+$")


def to_units(pairs: list) -> list[dict]:
    """
    분해의 [한국어, 영어] 쌍 → {"ko","en"}. 형식이 어긋난 항목은 버린다.
    대괄호·큰따옴표가 든 문자열은 JSON 구분자가 문자열 안으로 샌 붕괴 출력이라 버린다
    (스키마 패턴이 막지만, 패턴을 지원하지 않는 서버에 대비한 이중 방어).
    """
    out = []
    for p in pairs or []:
        if isinstance(p, (list, tuple)) and len(p) == 2 and str(p[0]).strip():
            ko, en = str(p[0]).strip(), str(p[1]).strip()
            if _BROKEN.isdisjoint(ko + en):
                out.append({"ko": ko, "en": en})
    return out


def all_candidates(units: list[dict], cats_list: list[set[int]], variant: str) -> list[list[dict]]:
    """
    조각별 후보: 어휘 일치 → 한국어 질의 → 영어 질의 순서로 합치고 중복 제거.
    모든 조각의 [한국어, 영어] 질의를 한 번에 임베딩한다(속도판, 조각마다 따로 하던 것을 묶음).
    """
    if not units:
        return []
    texts = [t for u in units for t in (u["ko"], u["en"])]
    cats = [c for c in cats_list for _ in (0, 1)]
    hits = search_mod.search_prefiltered_multi(texts, cats, variant, top_k=TOP_K)
    out = []
    for i, (u, c) in enumerate(zip(units, cats_list)):
        seen, merged = set(), []
        for h in search_mod.lexical_lookup(u["en"], variant, c) + hits[2 * i] + hits[2 * i + 1]:
            if h["tag"] not in seen:
                seen.add(h["tag"])
                merged.append(h)
        out.append(merged)
    return out


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
        # 압축 출력: 조각마다 후보 번호 배열, 또는 맞는 후보가 없을 때만 영어 구 문자열
        nums = {"type": "array", "items": {"type": "integer", "enum": list(range(len(cs))) or [0]},
                "minItems": 1, "maxItems": 3}
        phrase = {"type": "string", "pattern": PHRASE_PATTERN}
        props[f"u{i}"] = {"anyOf": [nums, phrase]} if cs else phrase
    schema = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

    notes: list[str] = []
    try:
        res = await _call_json(SYSTEM_SELECT, "\n".join(lines), schema, name, seed=LLM_SEED)
    except LLMError as e:
        logger.warning(f"선택 실패 → 분해 영어로 대체: {e}")
        notes.append(f"선택 실패: {e}")
        res = {}

    picks: list[list[str]] = []
    for i, (u, cs) in enumerate(zip(units, cands)):
        r = res.get(f"u{i}")
        if isinstance(r, list):
            tags = list(dict.fromkeys(cs[j]["tag"] for j in r if isinstance(j, int) and 0 <= j < len(cs)))
            picks.append(tags or ["phrase:" + u["en"]])
        elif isinstance(r, str) and r.strip():
            picks.append(["phrase:" + r.strip()])
        else:
            picks.append(["phrase:" + u["en"]])
            notes.append(f"{u['ko']}: 응답 없음 → 분해 영어 '{u['en']}'")
    return picks, notes


async def decompose(text: str) -> dict:
    """구조화 분해 1회 (JSON 스키마 강제)."""
    return await _call_json(SYSTEM_DECOMPOSE, text, SCHEMA_DECOMPOSE, "decompose", seed=LLM_SEED)


def assemble(picks: list[list[str]], all_cands: list[list[dict]], n_chars: int,
             variant: str) -> tuple[list[str], list[str], list[str]]:
    """
    선택 결과 → (캐릭터 태그, 본문 토큰, DB 밖이라 지운 태그). 앞의 n_chars 개 조각이 캐릭터 조각.
    캐릭터 조각은 cat4 태그만 남긴다(작품 cat3 은 최종에서 제외, 기존 정책).
    본문은 태그와 영어 구(쉼표 제거)가 섞이며, DB 실존 확인과 _(cosplay) 등 파생 태그 제거를 거친다.
    """
    tag_cat = {h["tag"]: h["category"] for cs in all_cands for h in cs}
    char_tags, body = [], []
    for i, ps in enumerate(picks):
        for p in ps:
            if p.startswith("phrase:"):
                t = _phrase_text(p)
                if t:
                    body.append(t)
            elif i < n_chars:
                if tag_cat.get(p) == 4 and p not in char_tags:
                    char_tags.append(p)
            else:
                body.append(p)
    db_tags = [t for t in body if t in tag_cat]
    _, dropped_db = search_mod.filter_existing_tags(db_tags, variant)
    dropped = set(dropped_db)
    body = [t for t in body if t not in dropped]
    body = search_mod.drop_character_derived(body)
    return char_tags, body, list(dropped_db)


async def describe(text: str, variant: str = MAIN_VARIANT, strip_names: list[str] | None = None) -> dict:
    """
    고급 모드의 묘사·배경 칸용: 분해 → 일반(cat0) 조각만 검색·선택 → 본문 토큰.
    분해의 characters 는 버린다(고급 모드는 캐릭터를 이름 칸으로 따로 찾는다).
    strip_names: 조각의 한국어 표기에서 이 이름들을 지운다(지우고 비면 조각을 버림) —
                 이름이 cat0 검색을 타면 'name_(cosplay)' 같은 파생 태그가 잡히기 때문.
    반환: {"tags": 본문 토큰, "hallucinated": DB 밖 태그, "people": {"girls","boys","others"}}
    """
    empty = {"tags": [], "hallucinated": [], "people": {"girls": 0, "boys": 0, "others": 0}}
    if not text.strip():
        return empty
    dec = await decompose(text)
    units = []
    for u in to_units(dec.get("units")):
        ko = search_mod.strip_names_from_text(u["ko"], strip_names).strip() if strip_names else u["ko"]
        if ko:
            units.append({"ko": " ".join(ko.split()), "en": u["en"]})
    people = {k: int(dec.get(k, 0) or 0) for k in ("girls", "boys", "others")}
    if not units:
        return {**empty, "people": people}
    cands = all_candidates(units, [set(search_mod.GENERAL_CATS)] * len(units), variant)
    picks, notes = await select(text, units, cands, variant)
    if notes:
        logger.info(f"structured(묘사) 선택 메모: {notes}")
    _, body, dropped = assemble(picks, cands, 0, variant)
    return {"tags": list(dict.fromkeys(body)), "hallucinated": dropped, "people": people}


async def generate_nl_or_phrase(
    korean_prompt: str,
    kept: list[str],
    en_units: list[str],
    nl_tone: str | None,
    variant: str,
    char_tags: list[str] | None = None,
    temperature: float | None = None,
) -> str:
    """
    자연어 단계 출력 분기: 일반 톤이면 자연어 문장, "phrase"면 단어형 보조출력.

    단어형(UI '단어형+')은 확정 태그(kept)가 표현 못 한 잔차만 짧은 영어 구로 낸다.
    중복 배제 기준이 kept 이므로 kept 를 그대로 넘기고, 태그의 의미 범위(definition)를
    배제맥락으로 함께 준다.

    톤별 입력(NL 의 목적은 'tag 수렴 과정에서 잘려나간 손실 복구'):
      - rich  : kept(최종 태그) 기반. 정합성 우선, 묘사적.
      - plain : en_units(DB 통과 전 영어 분해단위) 기반. kept 를 받으면 손실된 결과를
                받아 손실을 복구하라는 모순이 되므로, 수렴 전 정보를 입력으로 준다.
                캐릭터 정체성은 char_tags 를 앞에 보존해 SYSTEM_NL RULE2 의 'X from Y' 오프닝을 살린다.
                en_units 가 비면 kept 로 폴백.
    temperature: None 이면 톤별 권장 온도(llm.NL_TONES / PHRASE_TEMPERATURE). UI 슬라이더 값이 오면 그 값.
    """
    if nl_tone == "phrase":
        defs = search_mod.get_definitions(variant)
        return await llm.generate_phrase(korean_prompt, kept, defs=defs, temperature=temperature)
    if nl_tone == "plain":
        basis = en_units or kept
        if char_tags:
            basis = char_tags + [t for t in basis if t not in char_tags]
        return await llm.generate_nl_prompt(korean_prompt, basis, tone=nl_tone, temperature=temperature)
    # rich(및 기타): 최종 태그 기반. 태그가 비면 영어 분해단위로 폴백.
    nl_basis = kept or en_units
    return await llm.generate_nl_prompt(korean_prompt, nl_basis, tone=nl_tone, temperature=temperature)


def _phrase_text(p: str) -> str:
    """구 → 최종 프롬프트 토큰. UI 가 ', ' 로 태그를 나누므로 쉼표는 지운다."""
    return " ".join(p[len("phrase:"):].replace(",", " ").split())


async def stream_structured_pipeline(
    korean_prompt: str,
    variant: str = MAIN_VARIANT,
    generate_nl: bool = True,
    search_categories: set[int] | None = None,
    nl_tone: str | None = None,
    nl_temperature: float | None = None,
):
    """structured 를 단계별로 스트리밍. yield {"stage": korean|english|final|nl, "status": 표시 문구, "data": dict}."""
    unit_cats = set(search_categories) if search_categories else set(search_mod.GENERAL_CATS)

    # ── 1. 구조화 분해 ──
    dec = await decompose(korean_prompt)
    chars = to_units(dec.get("characters"))
    units = to_units(dec.get("units"))
    count = count_tags(dec)
    logger.info(f"structured 분해: 인원={count} 캐릭터={[c['ko'] for c in chars]} 조각={[u['ko'] + '/' + u['en'] for u in units]}")
    yield {"stage": "korean", "status": "한국어 키워드 추출 완료",
           "data": {"ko_units": [c["ko"] for c in chars] + [u["ko"] for u in units], "person_units": count}}
    en_units = [c["en"] for c in chars] + [u["en"] for u in units]
    yield {"stage": "english", "status": "영어 키워드 추출 완료", "data": {"en_units": en_units}}

    # ── 2. 조각별 후보 ──
    all_units = chars + units
    all_cands = all_candidates(all_units, [set(search_mod.CHARACTER_CATS)] * len(chars) + [unit_cats] * len(units),
                               variant)

    # ── 3. 선택 ──
    picks, notes = await select(korean_prompt, all_units, all_cands, variant)
    if notes:
        logger.info(f"structured 선택 메모: {notes}")

    # ── 4. 최종 조립 + 이중 방어 ──
    char_tags, body, dropped_db = assemble(picks, all_cands, len(chars), variant)
    final = list(dict.fromkeys(count + char_tags + body))
    yield {"stage": "final", "status": "태그 완성", "data": {"final_tags": final, "hallucinated": list(dropped_db)}}

    # ── 5. 자연어 (기존 톤 기능 그대로) ──
    nl_prompt = ""
    if generate_nl:
        nl_prompt = await generate_nl_or_phrase(korean_prompt, final, en_units, nl_tone, variant,
                                                char_tags=char_tags, temperature=nl_temperature)
    yield {"stage": "nl", "status": "완료", "data": {"nl_prompt": nl_prompt}}


async def run_structured_pipeline(korean_prompt: str, variant: str = MAIN_VARIANT, generate_nl: bool = False,
                          search_categories: set[int] | None = None, nl_tone: str | None = None) -> dict:
    """스트림을 모아 dict 로 반환 (검증·벤치마크용)."""
    out: dict = {}
    async for ev in stream_structured_pipeline(korean_prompt, variant, generate_nl, search_categories, nl_tone):
        out.update(ev["data"])
    return out
