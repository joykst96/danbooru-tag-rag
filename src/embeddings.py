"""
임베딩 모듈

텍스트를 1024차원 벡터로 변환한다.
multilingual-e5-large 모델을 사용하며, 로컬에 받아둔 경로에서 로드한다.

E5 모델 특성:
    - 비대칭 검색 모델. 문서(passage)와 질의(query)에 서로 다른 프리픽스를 붙인다.
    - normalize_embeddings=True 로 정규화하여 L2 거리 ↔ 코사인 유사도 변환을 단순화한다.
"""

import logging

from sentence_transformers import SentenceTransformer

from .config import (
    get_embedding_model_path,
    EMBEDDING_DEVICE,
    E5_QUERY_PREFIX,
    E5_PASSAGE_PREFIX,
    EMBED_BATCH_SIZE,
)

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    임베딩 모델 관리 (클래스 레벨 싱글톤).

    모델은 무거우므로 한 번만 로드하여 공유한다.
    """

    _model: SentenceTransformer | None = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """모델을 반환 (최초 호출 시 로드). 로컬 경로 우선, 없으면 HF 폴백."""
        if cls._model is None:
            model_path = get_embedding_model_path()
            logger.info(f"임베딩 모델 로드 중: {model_path} (device={EMBEDDING_DEVICE or 'auto'})")
            cls._model = SentenceTransformer(model_path, device=EMBEDDING_DEVICE)
            logger.info("임베딩 모델 로드 완료")
        return cls._model

    @classmethod
    def embed_query(cls, text: str) -> list[float]:
        """검색 질의 1건을 벡터로 변환 (query 프리픽스 부착)."""
        model = cls.get_model()
        embedding = model.encode(
            f"{E5_QUERY_PREFIX}{text}",
            normalize_embeddings=True,
        )
        return embedding.tolist()

    @classmethod
    def embed_queries(cls, texts: list[str]) -> list[list[float]]:
        """여러 질의를 한 번에 벡터로 변환 (배치)."""
        model = cls.get_model()
        prefixed = [f"{E5_QUERY_PREFIX}{t}" for t in texts]
        embeddings = model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    @classmethod
    def embed_passages(cls, texts: list[str], show_progress: bool = True) -> list[list[float]]:
        """
        문서(passage)들을 벡터로 변환 (인덱스 빌드용, passage 프리픽스 부착).
        대량 처리이므로 배치 + 진행바 옵션을 둔다.
        """
        model = cls.get_model()
        prefixed = [f"{E5_PASSAGE_PREFIX}{t}" for t in texts]
        embeddings = model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=show_progress,
        )
        return embeddings.tolist()
