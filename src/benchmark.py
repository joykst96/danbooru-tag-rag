"""
벤치마크 스크립트 (variant a/b/c 비교)

새 세션에서 인덱스 빌드 후 이 스크립트로 variant들을 비교한다.
두 가지 모드:

  1) raw   : LLM 없이 순수 벡터검색만. 쿼리 → 각 variant에서 top_k 결과를 나란히 출력.
             "어느 인덱스가 어떤 태그를 끌어오는가"를 날것으로 비교 (가장 빠르고 직관적).
  2) pipe  : 전체 2-pass 파이프라인을 variant 조합별로 실행 (LLM 포함, 느림).

사용법:
  python -m danbooru_rag.benchmark raw          # 순수 검색 비교
  python -m danbooru_rag.benchmark pipe         # 파이프라인 비교
  python -m danbooru_rag.benchmark raw "파란 머리 소녀" "비 오는 거리"   # 커스텀 쿼리

기본 쿼리셋은 아래 DEFAULT_QUERIES. 본인이 자주 쓰는 프롬프트로 교체 권장.
"""

import sys
import asyncio

from .config import INDEX_VARIANTS, DEFAULT_TOP_K
from . import search as search_mod
from .pipeline import run_pipeline, PipelineConfig


# 자주 쓰는 한국어 프롬프트로 교체하면 본인 사용 패턴에 맞는 비교가 된다.
DEFAULT_QUERIES = [
    "파란 머리 소녀",
    "비 오는 거리를 걷는 여자",
    "교복 입은 트윈테일 캐릭터",
    "검을 든 갑옷 기사",
    "벚꽃 아래 앉아있는 소년",
]


def run_raw(queries: list[str], top_k: int = DEFAULT_TOP_K) -> None:
    """LLM 없이 각 variant의 순수 검색 결과를 나란히 출력."""
    for q in queries:
        print("\n" + "=" * 70)
        print(f"쿼리: {q}")
        print("=" * 70)
        for variant in INDEX_VARIANTS:
            hits = search_mod.search_one(q, variant=variant, top_k=top_k)
            print(f"\n[variant {variant.upper()}]")
            if not hits.results:
                print("  (결과 없음)")
                continue
            for r in hits.results:
                alias_preview = ", ".join(r.aliases[:3]) if r.aliases else "-"
                print(f"  {r.score:.3f}  {r.tag:30s} "
                      f"[{r.major}/{r.minor}] cat={r.category}  ({alias_preview})")


async def run_pipe(queries: list[str]) -> None:
    """variant 조합별로 전체 파이프라인을 실행 비교."""
    # 비교할 (pass1, pass2) variant 조합. 필요에 따라 추가/수정.
    combos = [
        ("a", "a"),
        ("c", "a"),   # Pass1 한국어(C) → Pass2 전부(A)
        ("c", "b"),   # Pass1 한국어(C) → Pass2 영어+별칭(B)
    ]
    for q in queries:
        print("\n" + "=" * 70)
        print(f"쿼리: {q}")
        print("=" * 70)
        for p1, p2 in combos:
            cfg = PipelineConfig(pass1_variant=p1, pass2_variant=p2)
            result = await run_pipeline(q, cfg)
            print(f"\n[Pass1={p1.upper()} → Pass2={p2.upper()}]")
            print(f"  키워드     : {result.keywords}")
            print(f"  Pass1후보  : {result.pass1_tags}")
            print(f"  정제표현   : {result.refined}")
            print(f"  최종태그   : {result.final_tags}")
            if result.suspicious:
                print(f"  의심태그   : {result.suspicious}")


def main():
    args = sys.argv[1:]
    mode = args[0] if args else "raw"
    queries = args[1:] if len(args) > 1 else DEFAULT_QUERIES

    if mode == "raw":
        run_raw(queries)
    elif mode == "pipe":
        asyncio.run(run_pipe(queries))
    else:
        print(f"알 수 없는 모드: {mode} (raw | pipe)")
        sys.exit(1)


if __name__ == "__main__":
    main()
