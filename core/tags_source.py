"""
태그 데이터 소스 → danbooru-tags.csv 생성

HF dataset mengod-hf/danbooru-tags-kr (2026-11-04, MIT) 의 통합 CSV danbooru-tags-kr-2026-11-04.csv
(150,993행) 를 revision 고정으로 받아, 이 프로젝트가 쓰는 danbooru-tags.csv 를 만든다.
인덱스 빌더(core/builder.py)와 UI 태그 설명·자동완성(/danbooru-tags.csv)이 이 파일 하나를 함께 읽는다.

    python -m core.tags_source            # 루트 danbooru-tags.csv 생성 (post_count >= 50)
    python -m core.tags_source --min-count 30 --out other.csv

소스 형식은 danbooru-tags.csv 와 같다("[대 > 소] 정의. 키워드: ...") 다만 아래를 고쳐 쓴다:
  - 헤더 행이 없다 → name,category,post_count,description 헤더를 붙인다
  - 태그명이 공백 구분("long hair") → "_" 로 바꾼다
  - 키워드 맨 앞의 "<헤어스타일>" 같은 분류 라벨은 별칭이 아니므로 뺀다

2026-11-08 이전에는 태그 집합을 출처·날짜 기록이 없는 옛 danbooru-tags.csv(114,092개)에서 가져오고
정의·별칭만 이 소스로 바꿔 끼웠다(core/kr2.py). 옛 파일은 Danbooru 에서 이름이 바뀐 태그 약 6,600개를
옛 이름으로 갖고 있었고 새 캐릭터 약 4만 개가 없었다. 이미지 시험(tag-embedding 벤치 gen_test/e11_tag_rename,
anima-base-v1.0 / SilvermoonMix 2B, 이름 바뀐 20쌍)에서 새 이름 인식 18/20, 옛 이름 16~17/20 이고
옛 이름만 통하는 경우가 사실상 없어 소스 하나로 통일했다.
"""

import argparse
import csv
import os

KR2_REPO = "mengod-hf/danbooru-tags-kr"
KR2_REVISION = "2f134f25e001868d4ed08ffbc1011b8eaead8528"
KR2_FILE = "danbooru-tags-kr-2026-11-04.csv"
MIN_COUNT = 50   # 이전 CSV 와 같은 컷(114,092 → 이 소스로 116,325개)

csv.field_size_limit(10 * 1024 * 1024)


def source_path() -> str:
    """소스 CSV 경로. HF 캐시에 받아 두고 재사용(오프라인이면 캐시 사용)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(KR2_REPO, KR2_FILE, repo_type="dataset", revision=KR2_REVISION)


def _drop_labels(desc: str) -> str:
    """'키워드:' 뒤 쉼표 목록에서 <분류 라벨> 토큰을 뺀다. 남는 별칭이 없으면 '키워드:' 부분을 지운다."""
    head, sep, kw = desc.rpartition("키워드:")
    if not sep:
        return desc
    kept = [t.strip() for t in kw.split(",") if t.strip() and not (t.strip().startswith("<") and t.strip().endswith(">"))]
    return f"{head}{sep} {', '.join(kept)}" if kept else head.rstrip()


def iter_rows(min_count: int = MIN_COUNT):
    """(name, category, post_count, description) — 소스 순서(post_count 내림차순) 그대로."""
    seen = set()
    with open(source_path(), encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            name = row[0].strip().replace(" ", "_")
            try:
                cat, count = int(row[1]), int(row[2])
            except ValueError:
                continue
            if count < min_count or name in seen:
                continue
            seen.add(name)
            yield name, cat, count, _drop_labels(row[3] if len(row) > 3 else "")


def write_tags_csv(out: str, min_count: int = MIN_COUNT) -> int:
    tmp = out + ".tmp"
    n = 0
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "category", "post_count", "description"])
        for r in iter_rows(min_count):
            w.writerow(r)
            n += 1
    os.replace(tmp, out)
    return n


if __name__ == "__main__":
    from .config import TAGS_CSV_PATH

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(TAGS_CSV_PATH))
    ap.add_argument("--min-count", type=int, default=MIN_COUNT)
    a = ap.parse_args()
    print(f"{write_tags_csv(a.out, a.min_count)}개 태그 → {a.out}")
