"""
빌더 모듈

danbooru-tags.csv 를 읽어 3개 variant(a/b/c) 인덱스를 빌드한다.

흐름:
    1. CSV 로드 → parser 로 ParsedTag 리스트 생성 (한 번만)
    2. 작가(1)/메타(5) 카테고리 제외, MIN_FREQUENCY 미만 제외
    3. 각 variant 별로:
        - build_embed_text 로 임베딩 텍스트 생성
        - embed_passages 로 벡터화 (GPU)
        - DanbooruTag 레코드로 묶어 LanceDB 저장

CSV 파싱은 공유하고 임베딩만 variant별로 3번 → GPU면 전체 15분 내외.

사용법:
    python -m danbooru_rag.builder            # 전체 variant 빌드
    python -m danbooru_rag.builder a          # 특정 variant만
"""

import sys
import csv
import logging

from .config import (
    TAGS_CSV_PATH,
    MIN_FREQUENCY,
    EXCLUDED_CATEGORIES,
    INDEX_VARIANTS,
)
from .parser import parse_row, build_embed_text, ParsedTag
from .embeddings import EmbeddingManager
from .database import TagDatabase, DanbooruTag

csv.field_size_limit(10 * 1024 * 1024)  # 긴 description 대응

logger = logging.getLogger(__name__)


def load_and_parse() -> list[ParsedTag]:
    """CSV를 읽어 필터링 후 ParsedTag 리스트를 반환한다."""
    logger.info(f"CSV 로드: {TAGS_CSV_PATH}")

    parsed_tags: list[ParsedTag] = []
    excluded_cat = 0
    excluded_freq = 0

    with open(TAGS_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # 헤더 건너뛰기 (name,category,post_count,description)

        for row in reader:
            if len(row) < 3:
                continue

            tag = row[0].strip()
            if not tag:
                continue

            try:
                category = int(row[1])
                frequency = int(row[2])
            except ValueError:
                continue

            description = row[3] if len(row) > 3 else ""

            # 필터: 작가(1)/메타(5) 제외
            if category in EXCLUDED_CATEGORIES:
                excluded_cat += 1
                continue
            # 필터: 저빈도 제외 (CSV가 이미 50컷이라 사실상 통과)
            if frequency < MIN_FREQUENCY:
                excluded_freq += 1
                continue

            parsed_tags.append(parse_row(tag, category, frequency, description))

    logger.info(f"파싱 완료: {len(parsed_tags)}건 유지")
    logger.info(f"  제외 - 카테고리(작가/메타): {excluded_cat}건")
    logger.info(f"  제외 - 저빈도(<{MIN_FREQUENCY}): {excluded_freq}건")

    # 타입 분포 로깅 (빌드 결과 확인용)
    type_dist: dict[str, int] = {}
    for p in parsed_tags:
        type_dist[p.desc_type] = type_dist.get(p.desc_type, 0) + 1
    logger.info(f"  description 타입 분포: {type_dist}")

    return parsed_tags


def build_variant(variant: str, parsed_tags: list[ParsedTag]) -> None:
    """단일 variant 인덱스를 빌드한다."""
    logger.info("=" * 60)
    logger.info(f"variant '{variant}' 빌드 시작")

    # 1. 임베딩 텍스트 생성
    embed_texts = [build_embed_text(p, variant) for p in parsed_tags]

    # 2. 벡터화 (GPU 배치)
    logger.info(f"임베딩 생성 중... ({len(embed_texts)}건)")
    vectors = EmbeddingManager.embed_passages(embed_texts, show_progress=True)

    # 3. 레코드 조립
    records = [
        DanbooruTag(
            tag=p.tag,
            category=p.category,
            frequency=p.frequency,
            major=p.major or "",
            minor=p.minor or "",
            definition=p.definition,
            aliases=p.aliases,
            vector=vec,
        )
        for p, vec in zip(parsed_tags, vectors)
    ]

    # 4. 저장
    db = TagDatabase(variant)
    db.create_table(records)
    logger.info(f"variant '{variant}' 빌드 완료: {len(records)}건")


def build_all(variants: tuple[str, ...] = INDEX_VARIANTS) -> None:
    """지정한 variant들을 모두 빌드한다. CSV 파싱은 한 번만 공유."""
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    parsed_tags = load_and_parse()
    if not parsed_tags:
        raise RuntimeError("파싱된 태그가 없습니다. CSV 경로를 확인하세요.")

    # 모델 미리 로드 (variant 루프마다 재로드 방지)
    EmbeddingManager.get_model()

    for variant in variants:
        build_variant(variant, parsed_tags)

    logger.info("=" * 60)
    logger.info(f"전체 빌드 완료: {list(variants)}")


if __name__ == "__main__":
    # 인자로 특정 variant 지정 가능 (예: python -m danbooru_rag.builder a)
    if len(sys.argv) > 1:
        target = tuple(v for v in sys.argv[1:] if v in INDEX_VARIANTS)
        if not target:
            print(f"유효한 variant 없음. 선택지: {INDEX_VARIANTS}")
            sys.exit(1)
        build_all(target)
    else:
        build_all()
