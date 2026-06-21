"""
검색 모듈 (순수 벡터검색 빌딩블록)

LLM 없이 벡터 검색만 담당한다. variant(a/b/c)를 파라미터로 받아
어느 인덱스든 동일 인터페이스로 검색할 수 있다.

설계 의도:
    - LLM 오케스트레이션(2-pass 흐름)은 pipeline.py 가 담당.
    - 이 모듈은 단독으로 import 하여 데이터 탐색/벤치마크에 바로 쓸 수 있다.
    - DB 인스턴스는 variant별로 캐싱하여 재연결 비용을 줄인다.
"""

import logging
from dataclasses import dataclass

from .config import DEFAULT_TOP_K, DEFAULT_THRESHOLD, CATEGORY_LABELS
from .database import TagDatabase, SearchResult
from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)

# variant별 DB 인스턴스 캐시
_db_cache: dict[str, TagDatabase] = {}

# ── 2분할 카테고리 집합 (측정: 캐릭터/작품이 속성검색 오염) ──
GENERAL_CATS = {0}        # 일반 속성·인원수·행동·배경
CHARACTER_CATS = {3, 4}   # 작품(3) + 캐릭터(4)

# ── 분할입력(고급) 전용: 작품명/캐릭터명 칸을 각각 정확 카테고리로 좁혀 검색 ──
# 일반 파이프라인은 CHARACTER_CATS{3,4}를 한 덩어리로 쓰지만, 분할입력에서는
# 작품명 칸은 cat3, 캐릭터명 칸은 cat4 로 분리 입력되므로 각 칸을 해당
# 카테고리로만 질의해 교차오염(작품명이 캐릭터 태그를, 그 반대를)을 줄인다.
COPYRIGHT_CATS = {3}      # 작품/출처
CHAR_ONLY_CATS = {4}      # 캐릭터

# variant별 '실존 태그 집합' 캐시 (환각 차단 코드필터용)
_tagset_cache: dict[str, set[str]] = {}


def get_db(variant: str) -> TagDatabase:
    """variant에 해당하는 DB 인스턴스를 캐싱하여 반환."""
    if variant not in _db_cache:
        _db_cache[variant] = TagDatabase(variant)
    return _db_cache[variant]


def get_tagset(variant: str) -> set[str]:
    """
    variant DB에 실존하는 전체 태그명 집합. (1회 로드 후 캐시)

    환각 차단의 코드 보증용: 파이프라인 최종 태그를 이 집합에 대조해
    DB에 없는 태그(LLM 생성)를 걸러낸다. silver_hair 처럼 DB에 없는
    태그명을 LLM이 만들어 출력하는 사례가 실측됨.
    """
    if variant not in _tagset_cache:
        tbl = get_db(variant).table
        df = tbl.to_pandas()
        _tagset_cache[variant] = set(df["tag"].tolist())
        logger.info(f"태그집합 로드 (variant={variant}): {len(_tagset_cache[variant])}개")
    return _tagset_cache[variant]


def filter_existing_tags(tags: list[str], variant: str) -> tuple[list[str], list[str]]:
    """
    태그 리스트를 (실존, 환각)으로 분리. 순서 유지.

    Returns:
        (kept, dropped) — kept는 DB에 있는 태그, dropped는 없어서 버린 태그
    """
    tagset = get_tagset(variant)
    kept, dropped = [], []
    for t in tags:
        (kept if t in tagset else dropped).append(t)
    return kept, dropped


@dataclass
class KeywordHits:
    """키워드 1개에 대한 검색 결과 묶음."""
    keyword: str
    results: list[SearchResult]

    @property
    def top(self) -> SearchResult | None:
        return self.results[0] if self.results else None


def search_one(
    keyword: str,
    variant: str,
    top_k: int = DEFAULT_TOP_K,
    exclude_categories: set[int] | None = None,
    include_categories: set[int] | None = None,
) -> KeywordHits:
    """단일 키워드를 지정 variant DB에서 검색."""
    query_vector = EmbeddingManager.embed_query(keyword)
    results = get_db(variant).search(
        query_vector,
        top_k=top_k,
        exclude_categories=exclude_categories,
        include_categories=include_categories,
    )
    return KeywordHits(keyword=keyword, results=results)


def search_many(
    keywords: list[str],
    variant: str,
    top_k: int = DEFAULT_TOP_K,
    exclude_categories: set[int] | None = None,
    include_categories: set[int] | None = None,
) -> list[KeywordHits]:
    """
    여러 키워드를 배치 임베딩 후 각각 검색.
    임베딩은 한 번에(배치), 검색은 키워드별로 수행.
    """
    if not keywords:
        return []

    # 콘솔 로그: 어떤 카테고리 DB를 어떤 키워드로 조회하는지
    if include_categories:
        cats = "+".join(CATEGORY_LABELS.get(c, str(c)) for c in sorted(include_categories))
    elif exclude_categories:
        cats = "전체-" + "+".join(CATEGORY_LABELS.get(c, str(c)) for c in sorted(exclude_categories))
    else:
        cats = "전체"
    logger.info(f"🔍 벡터DB 조회: [{cats}] {keywords}")

    query_vectors = EmbeddingManager.embed_queries(keywords)
    db = get_db(variant)

    hits: list[KeywordHits] = []
    for kw, qv in zip(keywords, query_vectors):
        results = db.search(
            qv,
            top_k=top_k,
            exclude_categories=exclude_categories,
            include_categories=include_categories,
        )
        hits.append(KeywordHits(keyword=kw, results=results))
    return hits


# ---------------------------------------------------------------------------
# 2분할 검색 (일반 / 캐릭터+작품) + 캐릭터 미스 폴백
# ---------------------------------------------------------------------------
def search_general(
    keywords: list[str],
    variant: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[KeywordHits]:
    """일반(cat 0) DB만 검색. 속성/인원수/행동/배경 단위용."""
    return search_many(keywords, variant, top_k=top_k, include_categories=GENERAL_CATS)


def search_pool(
    keywords: list[str],
    variant: str,
    top_k: int = DEFAULT_TOP_K,
    categories: set[int] | None = None,
) -> list[KeywordHits]:
    """
    지정 카테고리 풀에서 검색. 기본 파이프라인의 '검색 풀 설정'(고급 옵션)용.

    categories=None 이면 일반(cat 0)만 검색해 기존 동작과 동일.
    사용자가 작가/작품/캐릭터/메타를 풀에 추가하면 그 카테고리까지 검색하지만,
    일반 외 카테고리는 속성검색을 오염시켜 출력 품질이 떨어질 수 있다(경고 대상).
    """
    cats = categories if categories else GENERAL_CATS
    return search_many(keywords, variant, top_k=top_k, include_categories=cats)


def search_copyright(
    keywords: list[str],
    variant: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[KeywordHits]:
    """작품/출처(cat 3) DB만 검색. 분할입력 '작품명' 칸용. 폴백 없음."""
    return search_many(keywords, variant, top_k=top_k, include_categories=COPYRIGHT_CATS)


def search_character_only(
    keywords: list[str],
    variant: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[KeywordHits]:
    """
    캐릭터(cat 4) DB만 검색. 분할입력 '캐릭터명' 칸용.

    일반 파이프라인의 search_character 와 달리 **폴백을 하지 않는다.**
    분할입력에서는 캐릭터명이 DB에 없으면(마이너/오리지널) 일반DB로 재질의하지
    않고, 호출측에서 점수컷으로 미스 판정 → 이름 문자열만 NL에 그대로 쓴다.
    (사용자 결정: 없는 캐릭터는 태그 검색 자체를 버리고 NL 이름 보존.)
    """
    return search_many(keywords, variant, top_k=top_k, include_categories=CHAR_ONLY_CATS)


def search_character(
    keywords: list[str],
    variant: str,
    top_k: int = DEFAULT_TOP_K,
    fallback_threshold: float = DEFAULT_THRESHOLD,
    fallback_variant: str | None = None,
) -> list[KeywordHits]:
    """
    캐릭터+작품(cat 3,4) DB 검색. 단위가 캐릭터명일 때 사용.

    폴백: 캐릭터 DB top score 가 fallback_threshold 미만이면
    (= DB에 그 캐릭터가 없음) 같은 키워드를 일반 DB로 재질의해서
    속성으로 표현되게 한다. 마이너 캐릭터가 통째로 날아가는 것 방지.

    Note:
        폴백 트리거(fallback_threshold) 값은 측정으로 보정해야 한다.
        cross-lingual score 가 0.8x 압축이라 절대값 컷이 위험할 수 있음.
    """
    fallback_variant = fallback_variant or variant
    char_hits = search_many(
        keywords, variant, top_k=top_k, include_categories=CHARACTER_CATS
    )

    miss_keywords = [
        h.keyword for h in char_hits
        if h.top is None or h.top.score < fallback_threshold
    ]
    if not miss_keywords:
        return char_hits

    # 미스난 키워드만 일반 DB로 폴백 재질의
    logger.info(f"캐릭터 미스 → 일반 폴백: {miss_keywords}")
    fb_hits = {
        h.keyword: h
        for h in search_general(miss_keywords, fallback_variant, top_k=top_k)
    }

    # 폴백 결과로 교체 (히트한 캐릭터는 유지)
    merged: list[KeywordHits] = []
    for h in char_hits:
        if h.keyword in fb_hits and (h.top is None or h.top.score < fallback_threshold):
            merged.append(fb_hits[h.keyword])
        else:
            merged.append(h)
    return merged


def collect_tags(
    hits: list[KeywordHits],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[SearchResult], list[SearchResult]]:
    """
    여러 KeywordHits에서 태그를 모아 (확정, 후보)로 분리.

    - score >= threshold          → 확정(confirmed)
    - threshold-0.08 <= score < threshold → 후보(candidate)
    중복 태그는 가장 높은 score 기준으로 1개만 유지.
    """
    best: dict[str, SearchResult] = {}
    for h in hits:
        for r in h.results:
            prev = best.get(r.tag)
            if prev is None or r.score > prev.score:
                best[r.tag] = r

    confirmed, candidate = [], []
    for r in best.values():
        if r.score >= threshold:
            confirmed.append(r)
        elif r.score >= threshold - 0.08:
            candidate.append(r)

    confirmed.sort(key=lambda x: x.score, reverse=True)
    candidate.sort(key=lambda x: x.score, reverse=True)
    return confirmed, candidate


def group_by_category(results: list[SearchResult]) -> dict[str, list[SearchResult]]:
    """검색 결과를 카테고리 라벨별로 그룹핑 (general/copyright/character)."""
    grouped: dict[str, list[SearchResult]] = {}
    for r in results:
        label = CATEGORY_LABELS.get(r.category, "unknown")
        grouped.setdefault(label, []).append(r)
    return grouped
