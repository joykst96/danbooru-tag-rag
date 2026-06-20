"""
데이터베이스 모듈

LanceDB 벡터 데이터베이스 연결과 검색을 담당한다.
임베딩 텍스트 구성이 다른 3개 variant(a/b/c)를 각각 별도 DB로 관리한다.

정규화된 벡터를 쓰므로 L2 거리를 코사인 유사도로 변환한다:
    cosine_similarity = 1 - L2_distance / 2   (정규화 벡터 전제)
"""

import logging

import lancedb
from pydantic import BaseModel, ConfigDict

from .config import get_lancedb_path, get_table_name
from .exceptions import DatabaseNotInitializedError, SearchError

logger = logging.getLogger(__name__)


class DanbooruTag(BaseModel):
    """
    DB에 저장되는 태그 레코드.

    원본 대비 추가된 필드: category, major, minor, definition, aliases.
    이 메타데이터로 검색 후 카테고리 필터링/라벨링/한국어 표시가 가능해진다.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tag: str                  # 영문 태그명
    category: int             # 0=일반,3=작품,4=캐릭터 (1,5는 빌드 시 제외됨)
    frequency: int            # post_count
    major: str = ""           # 한국어 대분류
    minor: str = ""           # 한국어 소분류
    definition: str = ""      # 한국어 정의
    aliases: list[str] = []   # 한국어 별칭
    vector: list[float]       # 임베딩 벡터 (1024차원)


class SearchResult(BaseModel):
    """검색 결과 1건 (vector 제외, score 추가)."""
    tag: str
    category: int
    frequency: int
    major: str = ""
    minor: str = ""
    definition: str = ""
    aliases: list[str] = []
    score: float              # 코사인 유사도 (0~1, 1이 가장 유사)


def distance_to_similarity(distance: float) -> float:
    """L2 거리 → 코사인 유사도 (정규화 벡터 전제). 0~1로 클램프."""
    similarity = 1 - distance / 2
    return max(0.0, min(1.0, similarity))


class TagDatabase:
    """
    하나의 variant 인덱스에 대한 LanceDB 관리.

    variant('a'|'b'|'c')별로 인스턴스를 따로 만들어 사용한다.
    """

    def __init__(self, variant: str):
        self.variant = variant
        self.db_path = get_lancedb_path(variant)
        self.table_name = get_table_name(variant)
        logger.info(f"LanceDB 연결 (variant={variant}): {self.db_path}")
        self.db = lancedb.connect(str(self.db_path))
        self._table = None

    def table_exists(self) -> bool:
        return self.table_name in self.db.table_names()

    @property
    def table(self):
        """테이블 지연 로드."""
        if self._table is None:
            if not self.table_exists():
                raise DatabaseNotInitializedError(
                    f"variant '{self.variant}' 인덱스가 없습니다. builder를 먼저 실행하세요."
                )
            self._table = self.db.open_table(self.table_name)
        return self._table

    def create_table(self, tags: list[DanbooruTag]) -> None:
        """태그 레코드로 테이블 생성 (기존 있으면 덮어씀)."""
        logger.info(f"테이블 '{self.table_name}' 생성 중 ({len(tags)}건)")
        data = [tag.model_dump() for tag in tags]
        self._table = self.db.create_table(
            self.table_name, data=data, mode="overwrite"
        )
        logger.info("테이블 생성 완료")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        exclude_categories: set[int] | None = None,
    ) -> list[SearchResult]:
        """
        벡터 유사도 검색.

        Args:
            query_vector: 질의 벡터 (1024차원)
            top_k: 반환 개수
            exclude_categories: 결과에서 제외할 category 집합 (선택)

        Note:
            빌드 단계에서 이미 작가(1)/메타(5)를 제외했으므로,
            여기 exclude_categories 는 추가적 런타임 필터링용(예: 캐릭터 분리 검색).
        """
        try:
            # 카테고리 필터가 있으면 LanceDB where 절로 미리 거른다.
            # 단 top_k 가 필터로 줄어들 수 있으니 넉넉히 가져온 뒤 자른다.
            fetch_k = top_k if not exclude_categories else top_k * 3
            results = (
                self.table
                .search(query_vector)
                .limit(fetch_k)
                .to_list()
            )

            out: list[SearchResult] = []
            for r in results:
                cat = r["category"]
                if exclude_categories and cat in exclude_categories:
                    continue
                out.append(SearchResult(
                    tag=r["tag"],
                    category=cat,
                    frequency=r["frequency"],
                    major=r.get("major", ""),
                    minor=r.get("minor", ""),
                    definition=r.get("definition", ""),
                    aliases=r.get("aliases", []),
                    score=round(distance_to_similarity(r.get("_distance", 0)), 4),
                ))
                if len(out) >= top_k:
                    break
            return out

        except DatabaseNotInitializedError:
            raise
        except Exception as e:
            raise SearchError(f"검색 실패: {e}")
