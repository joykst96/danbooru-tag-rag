"""
분할입력 파이프라인 (고급 모드)

기본 모드(pipeline_structured)는 러프한 한 문장을 통째로 받는다.
이 모듈은 **인물칸 N개 + 배경칸 1개** 구조의 고급 입력을 처리한다.

동기 (Anima 공식 NL 팁):
    "Name a character, then describe their basic appearance."
    다인물일수록 이름만 나열하면 모델이 인물을 헷갈린다 → 이름 + 외형묘사를
    인물 단위로 묶어 NL을 쓴다. 칸을 나누는 목적은 검색 편의가 아니라
    **NL에서의 인물 구분**이다.

칸 구조:
    인물칸 = { series(작품명, 옵션) + name(캐릭터명, 옵션) + desc(캐릭터묘사) }
    배경칸 = { desc(배경/공통요소) }  1개

각 칸 처리:
    - series : 있으면 cat3(작품)만 검색. 미스(점수컷 미달)면 버림.
    - name   : 있으면 cat4(캐릭터)만 검색, **폴백 없음**.
               히트 → 태그 사용 / 미스 → 태그 검색 버리고 이름 문자열만 NL 보존.
    - desc   : structured 분해 → 일반 조각만 cat0 검색 → 번호 선택 (pipeline_structured.describe).
               이름이 있으면 "이름: 묘사"로 분해해 성별을 추론하게 하고, 조각에서는 이름을 지운다.
    - 배경    : 같은 흐름. 인물 이름은 조각에서 지운다. 인원수는 쓰지 않는다.

인원수:
    인물칸 하나 = 한 명. 칸마다 분해의 girls/boys/others 로 성별을 정해(여 > 남 > 기타 순,
    셋 다 0 이면 성별 미상이라 세지 않음) 칸 수만큼 합산한 뒤 1girl/2girls/1boy... 로 맨 앞에 둔다.
    (이전 4회 분해 위임 때는 칸마다 1girl 이 나와 중복 제거로 두 소녀도 1girl 이 됐다.)

최종 태그는 평탄하게 합쳐 출력(Danbooru 태그는 인물 귀속 불가).
NL만 인물 단위로 묶어 generate_nl_multi 로 생성한다.
"""

import logging
import re
from dataclasses import dataclass, field

from . import search as search_mod
from . import llm
from .pipeline_structured import describe, count_tags, generate_nl_or_phrase, needs_no_humans
from .config import MAIN_VARIANT

logger = logging.getLogger(__name__)


# 캐릭터/작품 히트 판정 컷. cross-lingual score 가 0.8x 대역에 압축되므로
# DEFAULT_THRESHOLD(0.80) 를 그대로 쓰되, 실측으로 보정 대상(인수인계서 §5-3).


@dataclass
class CharBlock:
    """입력 인물칸 1개."""
    name: str = ""        # 캐릭터명(공백 가능)
    series: str = ""      # 작품명(공백 가능)
    desc: str = ""        # 캐릭터묘사(한국어)
    is_original: bool = False  # 오리지널 캐릭터: 작품/캐릭터 검색 skip, NL이 임의 이름 할당
    is_passthrough: bool = False  # 패스스루: DB에 없는 신규 캐릭터. 작품/캐릭터 검색 skip, 입력 이름을 NL 핸들로 그대로 사용


@dataclass
class CharResult:
    """인물칸 1개의 처리 결과."""
    name: str = ""
    series: str = ""
    desc: str = ""
    is_original: bool = False                               # 오리지널 캐릭터 플래그
    is_passthrough: bool = False                            # 패스스루(신규 캐릭터) 플래그
    series_tags: list[str] = field(default_factory=list)   # cat3 히트
    char_tags: list[str] = field(default_factory=list)     # cat4 히트
    attr_tags: list[str] = field(default_factory=list)     # cat0 속성(환각필터 후)
    name_unmatched: bool = False                            # 캐릭터명 DB 미스 → NL에만 이름
    gender: str | None = None                               # "girls"/"boys"/"others", 묘사로 못 정하면 None
    name_en: str = ""                                       # 이름의 영어 표기(분해 characters, 한글 없으면 입력 그대로)
    series_en: str = ""                                     # 작품의 영어 표기

    def nl_name(self) -> str:
        """NL 에 쓸 이름. 매칭에 실패한 한글 이름은 영어 표기로 바꾼다(e10 n=8: "주연 from 젠레스 존 제로")."""
        if self.name_unmatched and _HANGUL.search(self.name) and self.name_en:
            return self.name_en
        return self.name

    def nl_series(self) -> str:
        """NL 에 쓸 작품명. 작품 태그를 못 찾은 한글 작품명은 영어 표기로 바꾼다."""
        if not self.series_tags and _HANGUL.search(self.series) and self.series_en:
            return self.series_en
        return self.series

    def output_tags(self) -> list[str]:
        """Danbooru 태그 출력용. 작품(series) 태그 제외 — 작품은 NL/캐릭터추론 보조용."""
        out: list[str] = []
        for t in (*self.char_tags, *self.attr_tags):
            if t not in out:
                out.append(t)
        return out

    def nl_tags(self) -> list[str]:
        """NL 생성용. 작품 태그 포함(캐릭터 정체/시리즈 맥락을 NL에 살린다)."""
        out: list[str] = []
        for t in (*self.series_tags, *self.char_tags, *self.attr_tags):
            if t not in out:
                out.append(t)
        return out


_drop_character_derived = search_mod.drop_character_derived


def _gender(people: dict) -> str | None:
    """칸 하나의 분해 인원 → 그 인물의 성별 키. 여 > 남 > 기타 순, 셋 다 0 이면 None."""
    for k in ("girls", "boys", "others"):
        if people.get(k, 0) > 0:
            return k
    return None


async def _describe_attr(
    desc: str, variant: str,
    strip_names: list[str] | None = None, name_hint: str = "", series_hint: str = "",
) -> tuple[list[str], dict, list[dict]]:
    """
    캐릭터묘사/배경 텍스트 → (속성 토큰, 분해 인원, 분해 characters). structured 분해·검색·선택(describe)에 위임.

    strip_names: 조각에서 지울 이름(자기 칸 이름·작품, 배경칸이면 모든 인물 이름).
                 이름이 cat0 검색을 타면 'name_(cosplay)' 같은 파생 태그가 잡히는 오염을 막는다.
    name_hint  : 있으면 "이름: 묘사"로 분해해 이름으로 성별을 추론하게 한다(조각에서는 strip_names 로 지움).
                 묘사가 비어 있으면 "이름 (작품)"만 분해한다 — 이름만 있는 칸도 성별을 정해 인원수를 세기 위해서다
                 (2026-11-08 운영 로그 표본 e10: 이름만 있는 칸은 인원수 태그가 하나도 안 붙었다, n=1·3·6·13·15·18·24).
    """
    if desc.strip():
        text = f"{name_hint}: {desc}" if name_hint else desc
    elif name_hint.strip():
        text = f"{name_hint} ({series_hint})" if series_hint.strip() else name_hint
    else:
        return [], {}, []
    names = [n for n in (strip_names or []) if n and n.strip()]
    res = await describe(text, variant, strip_names=names or None)
    return _drop_character_derived(res["tags"]), res["people"], res.get("characters", [])


_HANGUL = re.compile(r"[가-힣]")


def _english_names(name: str, series: str, dchars: list[dict]) -> tuple[str, str]:
    """
    칸의 이름·작품 → 영어 표기 (분해 characters 의 [한국어, 영어] 쌍에서 찾는다. 이미 한글이 없으면 그대로).
    고급 모드 이름칸은 사용자가 적은 표기(대개 한국어)로만 검색해서, 별칭에 한국어 이름이 없는 캐릭터를 놓쳤다
    (e10 n=8 주연 → zhu_yuan, n=11 아리 → ahri_(league)). 분해는 이름의 영어 표기를 함께 낸다.
    """
    def find(s: str, other: str) -> str:
        if not s.strip():
            return ""
        if not _HANGUL.search(s):
            return s.strip()
        k, ko_other = search_mod.norm_name(s), search_mod.norm_name(other)
        for c in dchars:
            ck = search_mod.norm_name(c["ko"])
            if ck and (ck == k or (ck in k or k in ck) and ck != ko_other) and not _HANGUL.search(c["en"]):
                return c["en"]
        return ""
    return find(name, series), find(series, name)


def _char_candidates(name: str, series: str, name_en: str, variant: str, top_k: int) -> list[dict]:
    """
    캐릭터 후보 = 이름 정확 일치(별칭·태그 본체, 한국어·영어) + 벡터 검색(이름 / "이름 (작품)" / 영어 이름).
    벡터 검색은 cat4 where 사전 필터(search_prefiltered_multi). 예전 search_character_only 는 사후 필터라
    상위권을 다른 카테고리가 채우면 후보가 줄었다(작품칸 "니케"·"젠레스 존 제로"는 cat3 후보 0건).
    "이름 (작품)" 질의: "주연" top8 에 zhu_yuan 이 없지만 "주연 (젠레스 존 제로)"는 top1.
    """
    queries = [name] + ([f"{name} ({series})"] if series else [])
    if name_en and search_mod.norm_name(name_en) != search_mod.norm_name(name):
        queries.append(name_en)
    hits = search_mod.search_prefiltered_multi(queries, [set(search_mod.CHAR_ONLY_CATS)] * len(queries), variant, top_k=top_k)
    exact = search_mod.exact_name_lookup([name, name_en], variant, 4, limit=top_k)
    return _merge(exact, *hits)


def _series_candidates(series: str, series_en: str, variant: str, top_k: int) -> list[dict]:
    """작품 후보 = 이름 정확 일치 + cat3 사전 필터 벡터 검색(작품명, 영어 작품명)."""
    queries = [series] + ([series_en] if series_en and search_mod.norm_name(series_en) != search_mod.norm_name(series) else [])
    hits = search_mod.search_prefiltered_multi(queries, [set(search_mod.COPYRIGHT_CATS)] * len(queries), variant, top_k=top_k)
    exact = search_mod.exact_name_lookup([series, series_en], variant, 3, limit=top_k)
    return _merge(exact, *hits)


def _merge(*lists: list[dict]) -> list[dict]:
    seen, out = set(), []
    for lst in lists:
        for h in lst:
            if h["tag"] not in seen:
                seen.add(h["tag"])
                out.append(h)
    return out


def _name_keys(*names: str) -> set[str]:
    """이름·태그 → 중복 비교 키(정규화 이름과, 태그면 괄호 한정어를 뗀 본체)."""
    out = set()
    for n in names:
        if n and n.strip():
            out |= {search_mod.norm_name(n), search_mod.norm_name(search_mod.tag_base(n.strip().replace(" ", "_")))}
    return out - {""}


def _drop_name_tokens(tokens: list[str], keys: set[str]) -> list[str]:
    """속성 토큰 중 캐릭터 태그·이름·작품과 같은 것을 뺀다(sakura_miko 옆의 'sakura miko' 구 등)."""
    return [t for t in tokens if search_mod.norm_name(t) not in keys]


async def _process_character(block: CharBlock, variant: str, top_k: int) -> CharResult:
    res = CharResult(
        name=block.name.strip(), series=block.series.strip(),
        desc=block.desc.strip(), is_original=block.is_original,
        is_passthrough=block.is_passthrough,
    )

    # 패스스루(신규 캐릭터): DB에 아직 없는 캐릭터를 사용자가 직접 지정.
    #   작품/캐릭터 DB 검색·LLM 선택을 모두 skip 하고, 입력한 이름/작품을 그대로
    #   NL 핸들로 쓴다(생성 태그가 아니므로 output_tags 에는 안 들어가고 nl_tags 에만 반영).
    #   묘사(desc)는 평소대로 structured 분해·검색·선택을 태워 속성 태그를 만든다(이쪽은 RAG 통과).
    #   name_unmatched 와 구분: 미스가 아니라 의도된 skip 이므로 False 유지.
    if res.is_passthrough:
        # 캐릭터 묘사칸에 자기 이름/작품을 또 적으면 그게 cat0 일반검색을 타서
        # 'name_(cosplay)' 같은 캐릭터 파생 태그가 잡힌다(캐릭터 정체성은 별도 검색이 담당).
        # 패스스루는 cat4 검색을 안 하지만, 입력 이름이 묘사에 섞이면 오염은 동일하므로
        # 자기 칸 이름/작품을 조각에서 제거하고 파생 태그도 후처리로 거른다.
        res.attr_tags, people, _ = await _describe_attr(
            res.desc, variant, strip_names=[res.name, res.series], name_hint=res.name, series_hint=res.series,
        )
        res.attr_tags = _drop_name_tokens(res.attr_tags, _name_keys(res.name, res.series))
        res.gender = _gender(people)
        return res

    # 캐릭터묘사 → structured 분해·검색·선택. (캐릭터 선택보다 먼저 한다: 분해의 영어 이름을 후보 검색에 쓴다)
    # 자기 칸 이름/작품 오염 차단: 묘사칸에 자기 캐릭터 이름을 또 적으면(예: 묘사에
    # "A캐릭터가 ~한다") 그 이름이 cat0 일반검색을 타서 'A_(cosplay)' 같은 파생 태그가
    # 잡혀 최종에 [정식 캐릭터태그, A_(cosplay)] 가 중복으로 들어간다. 캐릭터 정체성은
    # 아래의 cat4 검색이 담당하므로, 묘사칸은 속성/동작만 찾으면 된다.
    # → 자기 name/series 를 조각에서 제거 + _(cosplay) 등 파생 태그 후처리 제거.
    # 오리지널 캐릭터는 이름 입력을 무시하므로 성별 추론에도 쓰지 않는다.
    res.attr_tags, people, dchars = await _describe_attr(
        res.desc, variant, strip_names=[res.name, res.series],
        name_hint="" if res.is_original else res.name,
        series_hint="" if res.is_original else res.series,
    )

    # 오리지널 캐릭터: 작품/캐릭터 DB 검색 안 함(이름/작품 입력도 무시). 묘사만 검색.
    if not res.is_original and (res.name or res.series):
        # 작품·캐릭터 후보군을 각각 cat3/cat4에서 받아서, LLM이 사용자 입력 기준으로 선택.
        # (벡터 top1 단순채택은 동명이인/동일작품 조연을 구분 못 함 → LLM 의미판단으로 교정.)
        res.name_en, res.series_en = _english_names(res.name, res.series, dchars)
        char_cands = _char_candidates(res.name, res.series, res.name_en, variant, top_k) if res.name else []
        series_cands = _series_candidates(res.series, res.series_en, variant, top_k) if res.series else []

        char_tag, series_tag = await llm.select_character_and_series(
            res.name, res.series, char_cands, series_cands,
        )
        if series_tag:
            res.series_tags = [series_tag]
        if res.name:
            if char_tag:
                res.char_tags = [char_tag]
            else:
                res.name_unmatched = True
                logger.info(f"캐릭터명 미스(이름만 NL): {res.name} (영어 {res.name_en!r})")
    # 캐릭터 태그·이름·작품과 같은 속성 토큰 제거 (e10 리허설 n=5: sakura_miko 옆에 'sakura miko' 구가 붙음)
    res.attr_tags = _drop_name_tokens(
        res.attr_tags, _name_keys(*res.char_tags, *res.series_tags, res.name, res.series))
    res.gender = _gender(people)
    return res


async def stream_split_pipeline(
    characters: list[CharBlock],
    background_desc: str = "",
    variant: str = MAIN_VARIANT,
    top_k: int = 5,
    generate_nl: bool = True,
    nl_tone: str | None = None,
    nl_temperature: float | None = None,
):
    """
    분할입력 파이프라인 스트리밍 (async generator).

    yield {"stage", "status", "data"}:
      - "characters" : 인물별 처리 결과(이름/태그/미스여부)
      - "final"      : 평탄 합본 태그
      - "nl"         : 인물 단위 자연어 프롬프트
    """
    char_results: list[CharResult] = []
    for idx, block in enumerate(characters, 1):
        if not (block.name.strip() or block.series.strip() or block.desc.strip()):
            continue
        cr = await _process_character(block, variant, top_k)
        char_results.append(cr)
        yield {
            "stage": "character",
            "status": f"인물 {idx} 처리 완료",
            "data": {
                "index": idx,
                "name": cr.name,
                "series": cr.series,
                "tags": cr.output_tags(),
                "name_unmatched": cr.name_unmatched,
                "is_passthrough": cr.is_passthrough,
            },
        }

    # 배경. 인물 이름은 검색에서 제거(공통칸 동작지시가 cosplay 등 오염 유발 방지)
    _names = [b.name for b in characters if b.name.strip()]
    background_tags, _, _ = await _describe_attr(background_desc, variant, strip_names=_names)
    background_tags = _drop_name_tokens(background_tags, _name_keys(*_names))
    yield {
        "stage": "background",
        "status": "배경 처리 완료",
        "data": {"background_tags": background_tags},
    }

    # 평탄 합본: 인원수(칸별 성별 합산) → 인물별 태그 → 배경
    people = {"girls": 0, "boys": 0, "others": 0}
    for c in char_results:
        if c.gender:
            people[c.gender] += 1
    flat: list[str] = list(count_tags(people))
    # 인물칸이 없는 배경만의 입력: 기본 모드와 같은 규칙으로 no_humans (pipeline_structured.needs_no_humans)
    if not char_results and "no_humans" not in background_tags and needs_no_humans(background_desc, background_tags, variant):
        flat.append("no_humans")
    for c in char_results:
        for t in c.output_tags():
            if t not in flat:
                flat.append(t)
    for t in background_tags:
        if t not in flat:
            flat.append(t)
    yield {
        "stage": "final",
        "status": "태그 완성",
        "data": {"final_tags": flat},
    }

    # 인물 단위 NL. 인물칸이 없으면(배경만) 기본 모드와 같은 단일 장면 생성기를 쓴다:
    # SYSTEM_NL_MULTI 는 CHARACTER 블록이 하나 이상이라고 전제해서, 배경만 주면 모델이
    # "Please provide the CHARACTER blocks ..." 같은 요청 문장을 내는 경우가 있었다(2026-11-08 실측).
    # 단어형+(phrase)도 기본 모드와 같은 구 생성기를 쓴다. generate_nl_multi 에는 phrase 톤이 없어서
    # 문장이 나왔고, UI 는 그 문장을 쉼표로 잘라 구 뱃지로 그렸다(2026-11-08 확인).
    nl_prompt = ""
    if generate_nl and (not char_results or nl_tone == "phrase"):
        intent = " / ".join([c.desc for c in char_results if c.desc] + ([background_desc.strip()] if background_desc.strip() else []))
        basis = flat  # 인물칸이 없으면 flat = (no_humans) + 배경 태그
        nl_prompt = await generate_nl_or_phrase(intent, basis, [], nl_tone, variant, temperature=nl_temperature)
    elif generate_nl:
        char_payload = [
            {"name": c.nl_name(), "series": c.nl_series(), "tags": c.nl_tags(), "desc": c.desc, "is_original": c.is_original, "is_passthrough": c.is_passthrough}
            for c in char_results
        ]
        nl_prompt = await llm.generate_nl_multi(
            char_payload,
            background_tags=background_tags,
            background_desc=background_desc.strip(),
            tone=nl_tone,
            temperature=nl_temperature,
        )
    yield {
        "stage": "nl",
        "status": "완료",
        "data": {"nl_prompt": nl_prompt},
    }
