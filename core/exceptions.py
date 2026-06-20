"""
예외 모듈

프로젝트 전용 예외 계층.
"""


class DanbooruRAGError(Exception):
    """프로젝트 기본 예외."""
    pass


class DatabaseNotInitializedError(DanbooruRAGError):
    """인덱스가 빌드되지 않은 상태에서 검색을 시도한 경우."""

    def __init__(self, message: str | None = None):
        super().__init__(
            message or "데이터베이스가 초기화되지 않았습니다. builder를 먼저 실행하세요."
        )


class SearchError(DanbooruRAGError):
    """검색 처리 실패."""
    pass


class LLMError(DanbooruRAGError):
    """LLM 호출 실패."""
    pass
