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
from dataclasses import dataclass, field

from . import search as search_mod
from . import llm
from .pipeline_structured import describe, count_tags, generate_nl_or_phrase
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
    strip_names: list[str] | None = None, name_hint: str = "",
) -> tuple[list[str], dict]:
    """
    캐릭터묘사/배경 텍스트 → (속성 토큰, 분해 인원). structured 분해·검색·선택(describe)에 위임.

    strip_names: 조각에서 지울 이름(자기 칸 이름·작품, 배경칸이면 모든 인물 이름).
                 이름이 cat0 검색을 타면 'name_(cosplay)' 같은 파생 태그가 잡히는 오염을 막는다.
    name_hint  : 있으면 "이름: 묘사"로 분해해 이름으로 성별을 추론하게 한다(조각에서는 strip_names 로 지움).
    """
    if not desc.strip():
        return [], {}
    text = f"{name_hint}: {desc}" if name_hint else desc
    names = [n for n in (strip_names or []) if n and n.strip()]
    res = await describe(text, variant, strip_names=names or None)
    return _drop_character_derived(res["tags"]), res["people"]


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
        res.attr_tags, people = await _describe_attr(
            res.desc, variant, strip_names=[res.name, res.series], name_hint=res.name,
        )
        res.gender = _gender(people)
        return res

    # 오리지널 캐릭터: 작품/캐릭터 DB 검색 안 함(이름/작품 입력도 무시). 묘사만 검색.
    if not res.is_original and (res.name or res.series):
        # 작품·캐릭터 후보군을 각각 cat3/cat4에서 받아서, LLM이 사용자 입력 기준으로 선택.
        # (벡터 top1 단순채택은 동명이인/동일작품 조연을 구분 못 함 → LLM 의미판단으로 교정.)
        def _cands(hits) -> list[dict]:
            out = []
            for r in (getattr(hits, "results", None) or []):
                out.append({
                    "tag": r.tag,
                    "score": r.score,
                    "aliases": getattr(r, "aliases", []) or [],
                })
            return out

        char_cands: list[dict] = []
        series_cands: list[dict] = []
        if res.name:
            c_hits = search_mod.search_character_only([res.name], variant, top_k=top_k)
            if c_hits:
                char_cands = _cands(c_hits[0])
        if res.series:
            s_hits = search_mod.search_copyright([res.series], variant, top_k=top_k)
            if s_hits:
                series_cands = _cands(s_hits[0])

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
                logger.info(f"캐릭터명 미스(이름만 NL): {res.name}")

    # 캐릭터묘사 → structured 분해·검색·선택.
    # 자기 칸 이름/작품 오염 차단: 묘사칸에 자기 캐릭터 이름을 또 적으면(예: 묘사에
    # "A캐릭터가 ~한다") 그 이름이 cat0 일반검색을 타서 'A_(cosplay)' 같은 파생 태그가
    # 잡혀 최종에 [정식 캐릭터태그, A_(cosplay)] 가 중복으로 들어간다. 캐릭터 정체성은
    # 위의 cat4(search_character_only) 검색이 담당하므로, 묘사칸은 속성/동작만 찾으면 된다.
    # → 자기 name/series 를 조각에서 제거 + _(cosplay) 등 파생 태그 후처리 제거.
    # 오리지널 캐릭터는 이름 입력을 무시하므로 성별 추론에도 쓰지 않는다.
    res.attr_tags, people = await _describe_attr(
        res.desc, variant, strip_names=[res.name, res.series],
        name_hint="" if res.is_original else res.name,
    )
    res.gender = _gender(people)
    return res


async def stream_split_pipeline(
    characters: list[CharBlock],
    background_desc: str = "",
    variant: str = MAIN_VARIANT,
    top_k: int = 5,
    generate_nl: bool = True,
    nl_tone: str | None = None,
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
    background_tags, _ = await _describe_attr(background_desc, variant, strip_names=_names)
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
    nl_prompt = ""
    if generate_nl and not char_results:
        nl_prompt = await generate_nl_or_phrase(background_desc.strip(), background_tags, [], nl_tone, variant)
    elif generate_nl:
        char_payload = [
            {"name": c.name, "series": c.series, "tags": c.nl_tags(), "desc": c.desc, "is_original": c.is_original, "is_passthrough": c.is_passthrough}
            for c in char_results
        ]
        nl_prompt = await llm.generate_nl_multi(
            char_payload,
            background_tags=background_tags,
            background_desc=background_desc.strip(),
            tone=nl_tone,
        )
    yield {
        "stage": "nl",
        "status": "완료",
        "data": {"nl_prompt": nl_prompt},
    }
