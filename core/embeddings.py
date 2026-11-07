"""
임베딩 모듈

텍스트를 1024차원 벡터로 변환한다.
multilingual-e5-large 모델을 사용하며, 로컬에 받아둔 경로에서 로드한다.

E5 모델 특성:
    - 비대칭 검색 모델. 문서(passage)와 질의(query)에 서로 다른 프리픽스를 붙인다.
    - normalize_embeddings=True 로 정규화하여 L2 거리 ↔ 코사인 유사도 변환을 단순화한다.
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import (
    get_embedding_model_path,
    EMBEDDING_DEVICE,
    E5_QUERY_PREFIX,
    E5_PASSAGE_PREFIX,
    EMBED_BATCH_SIZE,
    EMBEDDING_PROFILE,
    GEMMA2_QUERY_PROMPT,
    GEMMA2_PASSAGE_PROMPT,
    GEMMA2_CONFIG_KWARGS,
    GEMMA2_MAX_SEQ_LENGTH,
    GEMMA2_TOKEN_BUDGET,
)

logger = logging.getLogger(__name__)

_GEMMA2 = EMBEDDING_PROFILE == "gemma2"


class EmbeddingManager:
    """
    임베딩 모델 관리 (클래스 레벨 싱글톤).

    모델은 무거우므로 한 번만 로드하여 공유한다.
    EMBEDDING_PROFILE=gemma2 이면 embeddinggemma-2 (아래 _gemma2_* 경로),
    e5 이면 기존 e5-large 경로를 그대로 탄다(코드 변경 없음).
    """

    _model: SentenceTransformer | None = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """모델을 반환 (최초 호출 시 로드). 로컬 경로 우선, 없으면 HF 폴백."""
        if cls._model is None:
            model_path = get_embedding_model_path()
            logger.info(f"임베딩 모델 로드 중: {model_path} (device={EMBEDDING_DEVICE or 'auto'})")
            if _GEMMA2:
                cls._model = cls._gemma2_load(model_path)
            else:
                cls._model = SentenceTransformer(model_path, device=EMBEDDING_DEVICE)
            logger.info("임베딩 모델 로드 완료")
        return cls._model

    @classmethod
    def embed_query(cls, text: str) -> list[float]:
        """검색 질의 1건을 벡터로 변환 (query 프리픽스 부착)."""
        if _GEMMA2:
            return cls.get_model().encode(
                text, prompt_name=GEMMA2_QUERY_PROMPT, normalize_embeddings=True,
            ).tolist()
        model = cls.get_model()
        embedding = model.encode(
            f"{E5_QUERY_PREFIX}{text}",
            normalize_embeddings=True,
        )
        return embedding.tolist()

    @classmethod
    def embed_queries(cls, texts: list[str]) -> list[list[float]]:
        """여러 질의를 한 번에 벡터로 변환 (배치)."""
        if _GEMMA2:
            return cls.get_model().encode(
                list(texts), prompt_name=GEMMA2_QUERY_PROMPT, normalize_embeddings=True,
                batch_size=EMBED_BATCH_SIZE, show_progress_bar=False,
            ).tolist()
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

        gemma2: GEMMA2_PASSAGE_PROMPT(SentenceSimilarity) 를 붙여 인코딩 (쿼리와 같은 프롬프트 — 대칭 방식).
        """
        if _GEMMA2:
            return cls._gemma2_passages(texts, show_progress)
        model = cls.get_model()
        prefixed = [f"{E5_PASSAGE_PREFIX}{t}" for t in texts]
        embeddings = model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=show_progress,
        )
        return embeddings.tolist()

    # ------------------------------------------------------------------
    # embeddinggemma-2 전용 (tag-embedding 벤치 tagemb/embeddings.py 와 같은 로직)
    # ------------------------------------------------------------------
    @staticmethod
    def _gemma2_load(model_path: str) -> SentenceTransformer:
        """텍스트 인코더만, fp32 고정(float16 은 NaN), max_seq 8192 명시."""
        import torch

        model = SentenceTransformer(
            model_path,
            device=EMBEDDING_DEVICE,
            model_kwargs={"dtype": torch.float32},
            config_kwargs=GEMMA2_CONFIG_KWARGS,
        )
        model.max_seq_length = GEMMA2_MAX_SEQ_LENGTH
        return model

    @classmethod
    def _gemma2_passages(cls, texts: list[str], show_progress: bool) -> list[list[float]]:
        """
        토큰 예산 배치: (배치 크기 x 최장 토큰수) <= GEMMA2_TOKEN_BUDGET 이 되게 길이순으로 묶는다.
        고정 배치(256)로는 긴 정의가 한 배치에 몰려 VRAM 이 넘친다(벤치 실측 peak 19.2GB).
        패딩은 결과에 영향이 없다(벤치 리허설: 단건 vs 배치 cos=1.0).
        """
        model = cls.get_model()
        texts = list(texts)
        if not texts:
            return []
        max_seq = min(model.max_seq_length or GEMMA2_MAX_SEQ_LENGTH, GEMMA2_MAX_SEQ_LENGTH)
        prompt = model.prompts.get(GEMMA2_PASSAGE_PROMPT, "")
        enc = model.tokenizer([prompt + t for t in texts], add_special_tokens=True, truncation=False)
        lens = np.minimum([len(x) for x in enc["input_ids"]], max_seq)
        order = np.argsort(-lens, kind="stable")
        dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        out = np.empty((len(texts), dim_fn()), dtype=np.float32)
        try:
            from tqdm import tqdm
            bar = tqdm(total=len(texts), disable=not show_progress, desc="임베딩(토큰예산)")
        except ImportError:  # pragma: no cover
            bar = None
        i = 0
        while i < len(order):
            n = max(1, min(EMBED_BATCH_SIZE, GEMMA2_TOKEN_BUDGET // int(lens[order[i]])))
            idx = order[i:i + n].tolist()
            out[idx] = model.encode([texts[j] for j in idx], batch_size=len(idx), prompt_name=GEMMA2_PASSAGE_PROMPT,
                                    normalize_embeddings=True, show_progress_bar=False)
            if bar:
                bar.update(len(idx))
            i += n
        if bar:
            bar.close()
        return out.tolist()
