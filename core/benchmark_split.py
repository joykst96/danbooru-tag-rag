"""
2분할 DB 효과 측정.

측정에서 캐릭터/작품 태그가 속성검색을 오염시킴이 관측됨:
  은발적안 → chi_lian, mr._silvair (캐릭터가 색속성 1등)
  트윈테일 → twintail-chan, twintelle (캐릭터가 헤어스타일 밀어냄)
  적안     → chi_lian 1등, red_eyes는 후보에도 없음

이 스크립트는 같은 쿼리를 두 방식으로 검색해 나란히 출력한다:
  [무분할]   전체 DB (현행) — 캐릭터 섞임
  [일반전용] include={0} — 캐릭터 제거

기대: 일반전용에서 캐릭터(cat 3,4)가 빠지고 속성 태그가 위로 올라옴.
순수 속성 쿼리로 검증한다 (캐릭터를 의도하지 않은 쿼리).

사용법:
  python -m core.benchmark_split          # 기본 속성 쿼리셋
  python -m core.benchmark_split "은발"    # 커스텀
"""

import sys

from .config import DEFAULT_TOP_K
from . import search as S


# 속성을 의도한 쿼리 (캐릭터가 나오면 오염). raw/분해 측정서 오염 관측된 것 위주.
DEFAULT_QUERIES = [
    "은발",
    "적안",
    "트윈테일",
    "여캐",
    "물빛 머리",
    "청록색 눈동자",
]

VARIANT = "c"  # raw 결론: 한국어 의미는 C가 강함


def _fmt(r) -> str:
    cat_mark = {0: "일반", 3: "작품", 4: "캐릭터"}.get(r.category, str(r.category))
    return f"{r.score:.3f} [{cat_mark}] {r.tag}"


def run_one(query: str, top_k: int = DEFAULT_TOP_K) -> None:
    print("\n" + "=" * 70)
    print(f"쿼리: {query}")
    print("=" * 70)

    # [무분할] 전체 DB
    full = S.search_one(query, VARIANT, top_k=top_k)
    print(f"\n[무분할] 전체 DB (variant {VARIANT.upper()})")
    char_count_full = sum(1 for r in full.results if r.category in S.CHARACTER_CATS)
    for r in full.results:
        print(f"    {_fmt(r)}")
    print(f"    → 캐릭터/작품 비율: {char_count_full}/{len(full.results)}")

    # [일반전용] include={0}
    gen = S.search_one(query, VARIANT, top_k=top_k, include_categories=S.GENERAL_CATS)
    print(f"\n[일반전용] include={{0}}")
    for r in gen.results:
        print(f"    {_fmt(r)}")

    # 무분할 1등이 캐릭터였는데 일반전용에서 속성으로 바뀌었나
    full_top = full.top
    gen_top = gen.top
    if full_top and gen_top:
        if full_top.category in S.CHARACTER_CATS and gen_top.category == 0:
            print(f"\n    ★ 오염 제거: 1등 [{full_top.tag}](캐릭터) → [{gen_top.tag}](일반)")
        elif full_top.category == 0:
            print(f"\n    (1등이 원래 일반: {full_top.tag} — 오염 없던 케이스)")


def main():
    queries = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_QUERIES
    for q in queries:
        run_one(q)


if __name__ == "__main__":
    main()
