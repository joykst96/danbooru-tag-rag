"""
분해 경로 측정 스크립트 (한국어 의미단위 분해 효과 실측)

세 경로가 같은 입력에서 끌어오는 후보를 나란히 출력한다:

  [1] 통문장      : 한국어 통문장을 e5로 직접 질의 (raw 에서 이미 본 것)
  [2] 분해(KO)    : LLM이 한국어 의미단위로 쪼갬 → 각 한국어 조각을 직접 질의 ★미측정
  [3] LLM영어KW   : LLM이 한국어→영어 키워드 추출 → 영어 질의 (pipe 의 Pass1 방식)

비교축:
  [3] vs [2] = LLM 영어변환 수렴 우회 효과 (영어KW가 head로 뭉개는지)
  [1] vs [2] = 분해 자체 효과 (통문장 오염 vs 조각별 독립 질의)

인원수 단위는 분해 경로에서 따로 표시한다 (검색에 안 맡기는 게 핵심이므로).

사용법:
  python -m core.benchmark_decompose                # 기본 쿼리셋
  python -m core.benchmark_decompose "쿼리1" "쿼리2"  # 커스텀
"""

import sys
import asyncio

from .config import DEFAULT_TOP_K
from . import search as search_mod
from . import llm
from . import llm_decompose as dec


# 복합 디테일 입력 위주 (분해 실익이 있다면 여기서 드러남).
# 로그에서 관측된 '꼬리 디테일이 묻히는' 유형으로 구성.
DEFAULT_QUERIES = [
    "은발에 트윈테일 로리, 적안, 머리 양쪽으로 드래곤 뿔",
    "허리까지 오는 구도에 비 내리는 밤에 우산을 쓰고 걷는 울프컷 검정 중간 머리 군인",
    "여자 둘이 남자 하나를 둘러싸고 무표정하게 쳐다본다",
    "교복 입은 트윈테일 소녀가 창가에서 석양을 보며 책을 읽는다",
]


# Pass1 검색에 쓸 variant (raw 결론: 한국어 의미는 C가 강함)
KO_VARIANT = "c"
# 영어 키워드 검색에 쓸 variant (raw 결론: 정확단어는 B)
EN_VARIANT = "b"


async def run_one(query: str, top_k: int = DEFAULT_TOP_K) -> None:
    print("\n" + "=" * 74)
    print(f"쿼리: {query}")
    print("=" * 74)

    # ── [1] 통문장 직접 질의 ──
    hits_full = search_mod.search_one(query, variant=KO_VARIANT, top_k=top_k)
    print(f"\n[1] 통문장 직접질의 (variant {KO_VARIANT.upper()})")
    for r in hits_full.results[:top_k]:
        print(f"    {r.score:.3f}  {r.tag:30s} ({', '.join(r.aliases[:2]) or '-'})")

    # ── [2] 한국어 의미단위 분해 → 조각별 질의 ──
    units = await dec.decompose_korean(query)
    print(f"\n[2] 한국어 분해 → 조각별 질의 (variant {KO_VARIANT.upper()})")
    print(f"    분해 단위: {units}")
    person_units = [u for u in units if dec.looks_like_person_unit(u)]
    attr_units = [u for u in units if not dec.looks_like_person_unit(u)]
    if person_units:
        print(f"    └ 인원수 단위(검색 제외, 직접확정 대상): {person_units}")

    unit_candidates: dict[str, list[str]] = {}
    if attr_units:
        hits_units = search_mod.search_many(attr_units, variant=KO_VARIANT, top_k=5)
        for h in hits_units:
            cands = [r.tag for r in h.results[:5]]
            unit_candidates[h.keyword] = cands
            top = h.results[0] if h.results else None
            top_s = f"{top.score:.3f} {top.tag}" if top else "(없음)"
            print(f"      · {h.keyword:20s} → {top_s}   [{', '.join(cands[:4])}]")

    # 분해기반 선별
    selected = await dec.select_from_decomposed(unit_candidates)
    # 인원수는 분해 단위에서 직접 부여 (간단 매핑; 측정용)
    print(f"    선별 결과(속성): {selected}")
    if person_units:
        print(f"    + 인원수 직접확정: {person_units}  (← 검색 거치지 않음)")

    # ── [3] LLM 영어 키워드 추출 → 영어 질의 ──
    try:
        en_kws = await llm.extract_keywords(query)
    except Exception as e:
        en_kws = []
        print(f"\n[3] LLM 영어키워드 추출 실패: {e}")
    if en_kws:
        print(f"\n[3] LLM 영어키워드 → 질의 (variant {EN_VARIANT.upper()})")
        print(f"    영어 키워드: {en_kws}")
        hits_en = search_mod.search_many(en_kws, variant=EN_VARIANT, top_k=3)
        for h in hits_en:
            top = h.results[0] if h.results else None
            top_s = f"{top.score:.3f} {top.tag}" if top else "(없음)"
            print(f"      · {h.keyword:20s} → {top_s}")


async def main_async(queries: list[str]) -> None:
    for q in queries:
        await run_one(q)


def main():
    queries = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_QUERIES
    asyncio.run(main_async(queries))


if __name__ == "__main__":
    main()
