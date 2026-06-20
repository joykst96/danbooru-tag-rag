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


def get_db(variant: str) -> TagDatabase:
    """variant에 해당하는 DB 인스턴스를 캐싱하여 반환."""
    if variant not in _db_cache:
        _db_cache[variant] = TagDatabase(variant)
    return _db_cache[variant]


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
) -> KeywordHits:
    """단일 키워드를 지정 variant DB에서 검색."""
    query_vector = EmbeddingManager.embed_query(keyword)
    results = get_db(variant).search(
        query_vector, top_k=top_k, exclude_categories=exclude_categories
    )
    return KeywordHits(keyword=keyword, results=results)


def search_many(
    keywords: list[str],
    variant: str,
    top_k: int = DEFAULT_TOP_K,
    exclude_categories: set[int] | None = None,
) -> list[KeywordHits]:
    """
    여러 키워드를 배치 임베딩 후 각각 검색.
    임베딩은 한 번에(배치), 검색은 키워드별로 수행.
    """
    if not keywords:
        return []

    query_vectors = EmbeddingManager.embed_queries(keywords)
    db = get_db(variant)

    hits: list[KeywordHits] = []
    for kw, qv in zip(keywords, query_vectors):
        results = db.search(qv, top_k=top_k, exclude_categories=exclude_categories)
        hits.append(KeywordHits(keyword=kw, results=results))
    return hits


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
