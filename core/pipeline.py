"""
파이프라인 모듈 (2-pass cross-lingual 오케스트레이션)

search(벡터검색) + llm(호출)을 조립하여 전체 흐름을 실행한다.

흐름:
    [Pass 1] 한국어 입력 → (직접 cross-lingual 검색) → 거친 후보
       └ 보조적으로 LLM 키워드추출도 병행 가능 (설정)
    [Refine] 후보 태그를 LLM에 보여주고 영어 의도표현 정제
    [Pass 2] 정제된 영어 → 정밀 검색 → 정밀 후보
    [구조화] 카테고리별 그룹핑 (캐릭터/작품 분리)
    [Select] LLM이 구조화된 실존 후보 중 최종 선택
    [NL/Verify] 자연어 프롬프트 생성 + 의심태그 표시 (병렬)

모든 단계 파라미터(variant, top_k, threshold)는 인자로 노출하여
새 세션의 벤치마크/튜닝에서 자유롭게 조절한다.

step_callback 으로 각 단계 진행상황을 외부(api SSE)에 흘려보낸다.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from . import search as search_mod
from . import llm
from .config import (
    DEFAULT_VARIANT,
    DEFAULT_TOP_K,
    DEFAULT_THRESHOLD,
)

logger = logging.getLogger(__name__)

# 단계 진행 콜백 타입: (step_no, status_text, data) -> awaitable
StepCallback = Callable[[int, str, dict | None], Awaitable[None]]


@dataclass
class PipelineConfig:
    """2-pass 파이프라인 튜닝 파라미터. 벤치마크에서 이 값을 바꿔가며 비교."""
    # Pass 1 (한국어 직접 검색)
    pass1_variant: str = DEFAULT_VARIANT
    pass1_top_k: int = 5
    pass1_threshold: float = 0.78      # 한국어 cross-lingual은 점수가 낮게 나와 약간 완화
    use_pass1: bool = True             # False면 LLM 키워드추출만으로 시작

    # Pass 2 (영어 정밀 검색)
    pass2_variant: str = DEFAULT_VARIANT
    pass2_top_k: int = 8
    pass2_threshold: float = DEFAULT_THRESHOLD

    # 런타임 카테고리 제외 (빌드에서 작가/메타는 이미 빠짐; 필요 시 추가 제외)
    exclude_categories: set[int] = field(default_factory=set)

    # LLM 온도
    kw_temperature: float = 0.1
    refine_temperature: float = 0.1
    select_temperature: float = 0.1
    nl_temperature: float = 0.4

    # 자연어/검증 생성 여부
    generate_nl: bool = True
    run_verify: bool = True


@dataclass
class PipelineResult:
    """파이프라인 최종 산출물."""
    korean_prompt: str
    keywords: list[str]                      # LLM 1차 키워드
    pass1_tags: list[str]                     # Pass1에서 나온 후보 태그명
    refined: list[str]                        # 정제된 영어 표현
    final_tags: list[str]                     # 최종 선택 태그 (환각 필터 후)
    grouped: dict[str, list[str]]             # 카테고리별 구조화 후보
    nl_prompt: str = ""
    suspicious: list[str] = field(default_factory=list)
    hallucinated: list[str] = field(default_factory=list)  # DB에 없어 제거된 태그


async def _noop_callback(step: int, status: str, data: dict | None) -> None:
    pass


async def run_pipeline(
    korean_prompt: str,
    cfg: PipelineConfig | None = None,
    step_callback: StepCallback | None = None,
) -> PipelineResult:
    """2-pass 파이프라인 실행."""
    cfg = cfg or PipelineConfig()
    cb = step_callback or _noop_callback

    # -------------------------------------------------------------------
    # Step 1: LLM 키워드 추출 (+ 선택적 Pass1 한국어 직접 검색)
    # -------------------------------------------------------------------
    await cb(1, "키워드 추출 중...", None)
    keywords = await llm.extract_keywords(korean_prompt, cfg.kw_temperature)
    logger.info(f"키워드: {keywords}")

    pass1_tags: list[str] = []
    if cfg.use_pass1:
        # 한국어 원문을 의미단위로 쪼개 직접 검색하는 대신,
        # 1차로는 LLM 키워드 + 한국어 원문 자체를 함께 거친 그물로 던진다.
        await cb(2, "1차 DB 검색 중 (거친 그물)...", {"keywords": keywords})
        pass1_queries = keywords + [korean_prompt]
        hits1 = search_mod.search_many(
            pass1_queries,
            variant=cfg.pass1_variant,
            top_k=cfg.pass1_top_k,
            exclude_categories=cfg.exclude_categories or None,
        )
        confirmed1, candidate1 = search_mod.collect_tags(hits1, cfg.pass1_threshold)
        pass1_tags = [r.tag for r in (confirmed1 + candidate1)]
        logger.info(f"Pass1 후보: {pass1_tags}")

    # -------------------------------------------------------------------
    # Step 2: 후보를 보여주고 영어 의도표현 정제 (2-pass 핵심)
    # -------------------------------------------------------------------
    await cb(3, "후보 기반 의도 정제 중...", {"pass1_tags": pass1_tags})
    refine_input = pass1_tags if pass1_tags else keywords
    refined = await llm.refine_with_candidates(
        korean_prompt, refine_input, cfg.refine_temperature
    )
    if not refined:
        refined = keywords  # 정제 실패 시 1차 키워드로 폴백
    logger.info(f"정제된 표현: {refined}")

    # -------------------------------------------------------------------
    # Step 3: Pass 2 정밀 검색 + 카테고리 구조화
    # -------------------------------------------------------------------
    await cb(4, "2차 정밀 검색 중...", {"refined": refined})
    hits2 = search_mod.search_many(
        refined,
        variant=cfg.pass2_variant,
        top_k=cfg.pass2_top_k,
        exclude_categories=cfg.exclude_categories or None,
    )
    confirmed2, candidate2 = search_mod.collect_tags(hits2, cfg.pass2_threshold)
    all_candidates = confirmed2 + candidate2

    grouped_results = search_mod.group_by_category(all_candidates)
    grouped = {label: [r.tag for r in rs] for label, rs in grouped_results.items()}

    # -------------------------------------------------------------------
    # Step 4: LLM 최종 선택
    # -------------------------------------------------------------------
    await cb(5, "최종 태그 선택 중...", {"grouped": grouped})
    final_tags = await llm.select_final(korean_prompt, grouped, cfg.select_temperature)
    if not final_tags:
        final_tags = [r.tag for r in confirmed2]  # 선택 실패 시 확정 태그로 폴백

    # ── 환각 차단 (코드 보증) ──
    # LLM이 후보에 없는 태그(예: silver_hair — DB에 실존하지 않음)를 생성하는
    # 사례가 실측됨. 프롬프트 권고만으로는 못 막으므로, 최종 태그를 Pass2 DB의
    # 실존 태그 집합에 대조해 없는 것은 코드로 제거한다. HANDOFF의
    # "DB 통과 태그만 출력" 원칙을 LLM 선의가 아니라 코드로 강제.
    final_tags, hallucinated = search_mod.filter_existing_tags(
        final_tags, cfg.pass2_variant
    )
    if hallucinated:
        logger.warning(f"환각 태그 제거 (DB에 없음): {hallucinated}")
    logger.info(f"최종 태그: {final_tags}")

    # -------------------------------------------------------------------
    # Step 5: 자연어 프롬프트 + 검증 (병렬)
    # -------------------------------------------------------------------
    await cb(6, "자연어 프롬프트 생성 및 검증 중...", {"final_tags": final_tags})
    nl_task = (
        llm.generate_nl_prompt(korean_prompt, final_tags, cfg.nl_temperature)
        if cfg.generate_nl else _empty_str()
    )
    verify_task = (
        llm.verify_tags(korean_prompt, final_tags, cfg.select_temperature)
        if cfg.run_verify else _empty_list()
    )
    nl_prompt, suspicious = await asyncio.gather(nl_task, verify_task)

    return PipelineResult(
        korean_prompt=korean_prompt,
        keywords=keywords,
        pass1_tags=pass1_tags,
        refined=refined,
        final_tags=final_tags,
        grouped=grouped,
        nl_prompt=nl_prompt,
        suspicious=suspicious,
        hallucinated=hallucinated,
    )


async def _empty_str() -> str:
    return ""


async def _empty_list() -> list[str]:
    return []
