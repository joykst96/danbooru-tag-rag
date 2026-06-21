"""
분할입력 파이프라인 (고급 모드)

기존 stream_decomposed_pipeline 은 러프한 한 문장을 통째로 받는다(기본 모드).
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
    - desc   : 통번역 분해 → cat0(일반) 속성 검색 → 환각필터.
    - 배경    : 통번역 분해 → cat0 속성 검색 → 환각필터. 인원수/캐릭터 보호 불필요.

인원수:
    별도 태그 강제주입 안 함. 인물칸 수만큼 NL이 인물 블록을 쓰므로
    인원수는 NL 구조로 자연스럽게 표현된다(학습된 캐릭터/행위묘사가 보조).

최종 태그는 평탄하게 합쳐 출력(Danbooru 태그는 인물 귀속 불가).
NL만 인물 단위로 묶어 generate_nl_multi 로 생성한다.
"""

import logging
import re
from dataclasses import dataclass, field

from . import search as search_mod
from . import llm
from .pipeline_decomposed import run_decomposed_pipeline
from .config import DEFAULT_THRESHOLD

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


@dataclass
class CharResult:
    """인물칸 1개의 처리 결과."""
    name: str = ""
    series: str = ""
    desc: str = ""
    is_original: bool = False                               # 오리지널 캐릭터 플래그
    series_tags: list[str] = field(default_factory=list)   # cat3 히트
    char_tags: list[str] = field(default_factory=list)     # cat4 히트
    attr_tags: list[str] = field(default_factory=list)     # cat0 속성(환각필터 후)
    name_unmatched: bool = False                            # 캐릭터명 DB 미스 → NL에만 이름

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


@dataclass
class SplitResult:
    characters: list[CharResult] = field(default_factory=list)
    background_tags: list[str] = field(default_factory=list)
    background_desc: str = ""
    final_tags: list[str] = field(default_factory=list)    # 전체 평탄 합본
    hallucinated: list[str] = field(default_factory=list)
    nl_prompt: str = ""


def _strip_names_from_text(text: str, names: list[str]) -> str:
    """
    배경 묘사에서 인물 이름을 제거한다(검색용 텍스트 전처리).

    공통칸에 "A와 B가 마주본다"처럼 인물 이름으로 동작을 지시하면, 그 이름이
    cat0 일반검색을 타서 캐릭터 파생 태그(예: xxx_(cosplay))를 잡는 오염이 생긴다.
    이름→인물 연결은 인물칸이 이미 담당하므로, 배경 '검색'에서는 이름을 빼고
    동작/구도만 찾는다. (NL용 배경 묘사는 원본을 그대로 쓰므로 구도 정보는 보존됨.)

    표기 변형(띄어쓰기) 대응: "라이덴 쇼군"으로 등록해도 배경에 "라이덴쇼군"으로
    붙여 쓴 경우를 잡기 위해, 이름의 공백을 '공백 0개 이상' 패턴으로 바꿔 매칭한다.
    """
    out = text
    # 긴 이름부터 제거(부분문자열 충돌 방지). 대소문자 무시.
    for nm in sorted([n.strip() for n in names if n.strip()], key=len, reverse=True):
        # 이름 내부 공백을 \s* 로 바꿔 띄어쓰기 변형 흡수("라이덴 쇼군"↔"라이덴쇼군")
        pat = r"\s*".join(re.escape(part) for part in nm.split())
        out = re.sub(pat, " ", out, flags=re.IGNORECASE)
    return out


# 배경 검색 결과에서 제거할 캐릭터 파생 태그 패턴.
# 배경은 cat0(일반)만 검색하지만, 이름이 새어들어가면 'xxx_(cosplay)' 같은
# 캐릭터 파생이 잡힌다. 배경 묘사 결과로서는 사실상 항상 오염이므로 후처리로 제거.
_COSPLAY_RE = re.compile(r"\(cosplay\)\s*$", re.IGNORECASE)


def _drop_character_derived(tags: list[str]) -> list[str]:
    """배경 결과에서 _(cosplay) 등 캐릭터 파생 태그를 제거(이중 방어)."""
    return [t for t in tags if not _COSPLAY_RE.search(t)]


async def _describe_attr(
    desc: str, variant: str, top_k: int,
    strip_names: list[str] | None = None, drop_derived: bool = False,
) -> tuple[list[str], list[str]]:
    """
    캐릭터묘사/배경 텍스트를 **검증된 4-step 본선(run_decomposed_pipeline)** 에 위임.

    한국어 분해 검색 + 통번역 분해 검색(한/영 후보풀 합본) + complete_tags LLM 선별
    + 환각 삼중필터를 그대로 받는다. (이전의 영어전용·선별없음 방식은 환각이 심해 폐기.)
    NL 은 인물 단위로 따로 만들므로 여기선 generate_nl=False.

    strip_names: 주어지면 검색 전 desc 에서 해당 이름들을 제거(배경칸 오염 방지).

    Returns: (final_tags, hallucinated)
    """
    if not desc.strip():
        return [], []
    search_text = _strip_names_from_text(desc, strip_names) if strip_names else desc
    if not search_text.strip():
        return [], []
    sub = await run_decomposed_pipeline(
        search_text,
        general_variant=variant,
        en_variant=variant,
        top_k=top_k,
        generate_nl=False,
    )
    tags = sub.final_tags
    if drop_derived:
        tags = _drop_character_derived(tags)
    return tags, sub.hallucinated


async def _process_character(block: CharBlock, variant: str, top_k: int) -> CharResult:
    res = CharResult(
        name=block.name.strip(), series=block.series.strip(),
        desc=block.desc.strip(), is_original=block.is_original,
    )

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

    # 캐릭터묘사 → 4-step 본선 위임(한/영 분해 + LLM 선별 + 환각필터)
    res.attr_tags, _ = await _describe_attr(res.desc, variant, top_k)
    return res


async def run_split_pipeline(
    characters: list[CharBlock],
    background_desc: str = "",
    variant: str = "b",
    top_k: int = 5,
    generate_nl: bool = True,
) -> SplitResult:
    """분할입력(고급) 파이프라인 실행."""
    res = SplitResult(background_desc=background_desc.strip())

    for block in characters:
        if not (block.name.strip() or block.series.strip() or block.desc.strip()):
            continue  # 빈 칸 skip
        res.characters.append(await _process_character(block, variant, top_k))

    # 배경칸 (cat0). 인물 이름은 검색에서 제거(공통칸 동작지시가 cosplay 등 오염 유발 방지)
    _names = [b.name for b in characters if b.name.strip()]
    res.background_tags, _ = await _describe_attr(
        background_desc, variant, top_k, strip_names=_names, drop_derived=True
    )

    # 전체 평탄 합본
    flat: list[str] = []
    for c in res.characters:
        for t in c.output_tags():
            if t not in flat:
                flat.append(t)
    for t in res.background_tags:
        if t not in flat:
            flat.append(t)
    res.final_tags = flat

    # 인물 단위 NL
    if generate_nl:
        char_payload = [
            {"name": c.name, "series": c.series, "tags": c.nl_tags(), "desc": c.desc, "is_original": c.is_original}
            for c in res.characters
        ]
        res.nl_prompt = await llm.generate_nl_multi(
            char_payload,
            background_tags=res.background_tags,
            background_desc=res.background_desc,
        )

    return res


async def stream_split_pipeline(
    characters: list[CharBlock],
    background_desc: str = "",
    variant: str = "b",
    top_k: int = 5,
    generate_nl: bool = True,
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
            },
        }

    # 배경. 인물 이름은 검색에서 제거(공통칸 동작지시가 cosplay 등 오염 유발 방지)
    _names = [b.name for b in characters if b.name.strip()]
    background_tags, _ = await _describe_attr(
        background_desc, variant, top_k, strip_names=_names, drop_derived=True
    )
    yield {
        "stage": "background",
        "status": "배경 처리 완료",
        "data": {"background_tags": background_tags},
    }

    # 평탄 합본
    flat: list[str] = []
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

    # 인물 단위 NL
    nl_prompt = ""
    if generate_nl:
        char_payload = [
            {"name": c.name, "series": c.series, "tags": c.nl_tags(), "desc": c.desc, "is_original": c.is_original}
            for c in char_results
        ]
        nl_prompt = await llm.generate_nl_multi(
            char_payload,
            background_tags=background_tags,
            background_desc=background_desc.strip(),
        )
    yield {
        "stage": "nl",
        "status": "완료",
        "data": {"nl_prompt": nl_prompt},
    }
