"""
새 한국어 소스 (nt 인덱스용)

HF dataset mengod-hf/danbooru-tags-kr (2026-11-04, MIT) 의 통합 CSV
danbooru-tags-kr-2026-11-04.csv (150,993행) 를 revision 고정으로 받는다.
(카테고리별 분할 CSV 5개는 이 통합 파일을 정확히 나눈 사본이라 쓰지 않는다 — 벤치에서 행 단위로 확인.)

형식은 danbooru-tags.csv 와 같다("[대 > 소] 정의. 키워드: ...") 다만:
  - 헤더 행이 없다
  - 태그명이 공백 구분("long hair") → "_" 로 바꿔 기존 태그와 결합
  - 키워드 맨 앞의 "<헤어스타일>" 같은 분류 라벨은 별칭이 아니므로 제외

태그 집합은 danbooru-tags.csv 그대로 두고, 각 태그의 정의·별칭·대소분류만 이 소스 값으로 바꾼다.
이 소스에 없는 태그는 정의·별칭 없이 이름만 남는다.
커버리지(벤치 실측): 일반 태그 한국어 정의 78% → 96%, 기존에 비어 있던 태그 20,313개 채워짐.

출처: tag-embedding 벤치 tagemb/kr2.py (같은 로직).
"""

import csv
from dataclasses import replace
from functools import lru_cache

from .parser import ParsedTag, parse_row

KR2_REPO = "mengod-hf/danbooru-tags-kr"
KR2_REVISION = "2f134f25e001868d4ed08ffbc1011b8eaead8528"
KR2_FILE = "danbooru-tags-kr-2026-11-04.csv"

csv.field_size_limit(10 * 1024 * 1024)


@lru_cache(maxsize=1)
def load_kr2() -> dict[str, ParsedTag]:
    """새 소스를 {태그명: ParsedTag} 로. HF 캐시에 받아 두고 재사용(오프라인이면 캐시 사용)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(KR2_REPO, KR2_FILE, repo_type="dataset", revision=KR2_REVISION)
    out: dict[str, ParsedTag] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            tag = row[0].strip().replace(" ", "_")
            try:
                p = parse_row(tag, int(row[1]), int(row[2]), row[3] if len(row) > 3 else "")
            except ValueError:
                continue
            p.aliases = [a for a in p.aliases if not (a.startswith("<") and a.endswith(">"))]
            out[tag] = p
    return out


def swap_source(p: ParsedTag, kr2: dict[str, ParsedTag]) -> ParsedTag:
    """기존 태그 p 의 한국어 정보(정의·별칭·계층)를 새 소스 값으로 바꾼 사본. 없으면 비운다."""
    n = kr2.get(p.tag)
    if n is None:
        return replace(p, desc_type="empty", major=None, minor=None, definition="", aliases=[])
    return replace(p, desc_type=n.desc_type, major=n.major, minor=n.minor,
                   definition=n.definition, aliases=list(n.aliases))


def build_embed_text_nt(p: ParsedTag) -> str:
    """
    nt 문서 텍스트 (embeddinggemma-2 모델 카드의 title 형식).
    p 는 swap_source 를 거친 태그. 정의·별칭이 둘 다 없으면 "title: none | text: {태그명}".
    """
    body = " ".join(x for x in [p.definition, " ".join(p.aliases)] if x).strip()
    if not body:
        return f"title: none | text: {p.tag_readable}"
    return f"title: {p.tag_readable} | text: {body}"
