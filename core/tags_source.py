"""
태그 데이터 소스 → danbooru-tags.csv 생성

HF dataset mengod-hf/danbooru-tags-kr (MIT) 의 통합 CSV 를 판(edition)별 revision 고정으로 받아,
이 프로젝트가 쓰는 danbooru-tags.csv 를 만든다 (KR2_EDITIONS, 기본 KR2_EDITION).
  - 2026-11-10판 (기본, 2026-11-10 교체): rev aa899ee, danbooru-tags-kr-2026-11-10.csv (152,801행 → post_count>=50 은 118,133개)
  - 2026-11-04판: rev 2f134f2, danbooru-tags-kr-2026-11-04.csv (150,993행 → 116,325개). --edition 2026-11-04 로 계속 만들 수 있다.
인덱스 빌더(core/builder.py)와 UI 태그 설명·자동완성(/danbooru-tags.csv)이 이 파일 하나를 함께 읽는다.

    python -m core.tags_source            # 루트 danbooru-tags.csv 생성 (post_count >= 50)
    python -m core.tags_source --min-count 30 --out other.csv
    python -m core.tags_source --edition 2026-11-04 --out old.csv

공통으로 아래를 고쳐 쓴다:
  - 헤더 행이 없다 → name,category,post_count,description 헤더를 붙인다
  - 태그명이 공백 구분("long hair") → "_" 로 바꾼다
  - 분류 라벨은 별칭이 아니므로 뺀다
출력 description 은 판과 상관없이 "[대 > 소] 정의. 키워드: 별칭1, 별칭2" 형식이다 (core/parser.py 는 그대로).

11-04판은 형식이 같고("키워드: <헤어스타일>, 별칭…") 라벨이 꺾쇠로 표시돼 있어 꺾쇠 토큰을 바로 뺀다 (_drop_labels).

11-10판은 형식이 "[대 > 소] 정의. <키워드> 별칭1, 별칭2." 로 바뀌었다 (끝 마침표는 형식이 붙인 것). 분류 라벨은
범용 라벨(<캐릭터> 등)이 지워진 행과, 구체적 라벨이 꺾쇠만 빠진 채 별칭으로 남은 행(1girl "인원, 여자 1명, …",
deadpool "마블캐릭터, 데드풀")으로 갈려서 새 형식만으로는 라벨과 별칭을 구분할 수 없다. 그래서
  - 11-04판을 라벨 기준판으로 같이 읽어, 그때 <라벨>이던 이름이 별칭에 기준판의 진짜 별칭보다 많이 나오면
    첫 번째 것을 뺀다 (chocolate "초콜릿, 초콜릿." → 진짜 별칭 "초콜릿" 은 남는다).
  - 마지막 별칭이 원래 "."로 끝나던 행("柑咲 etc.")은 형식 마침표와 구분이 안 된다. 기준판 마지막 별칭이
    "새 마지막 별칭 + ." 이면 기준판 값으로 되돌린다.
  - 11-04판에 없는 신규 1,808태그는 기준판이 없어 별칭을 그대로 쓴다 (대부분 작가 활동명이라 라벨이 거의 없다).
같은 규칙을 tag-embedding 벤치 tagemb/kr2.py 에 먼저 넣고 공통 150,993태그에서 11-04 결과와 대조했다
(의도하지 않은 차이 0, 벤치 README "두 번째 한국어 소스 11-10판 교체"). 이 파일의 출력도 벤치와 별칭이 같은지 대조했다.

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
# 판 → 고정 리비전·파일. label_ref: 라벨을 꺾쇠 없이 남긴 판이 라벨 이름을 가져올 기준판 (None 이면 꺾쇠 토큰을 바로 뺀다)
KR2_EDITIONS: dict[str, dict] = {
    "2026-11-04": {"revision": "2f134f25e001868d4ed08ffbc1011b8eaead8528",
                   "file": "danbooru-tags-kr-2026-11-04.csv", "label_ref": None},
    # main 최신(11-10 README 수정). 데이터를 올린 커밋은 76c33573506eb08f1571ac7550e59c3379681617
    "2026-11-10": {"revision": "aa899ee834557fa5e78f64b31e1e3d8f360a929a",
                   "file": "danbooru-tags-kr-2026-11-10.csv", "label_ref": "2026-11-04"},
}
KR2_EDITION = "2026-11-10"
KR2_REVISION = KR2_EDITIONS[KR2_EDITION]["revision"]
KR2_FILE = KR2_EDITIONS[KR2_EDITION]["file"]
MIN_COUNT = 50   # 이전 CSV 와 같은 컷(114,092 → 11-04판 116,325개 → 11-10판 118,133개)
NEW_KEYWORD_TAG = "<키워드>"  # 11-10판 형식의 별칭 표지

csv.field_size_limit(10 * 1024 * 1024)


def source_path(edition: str = KR2_EDITION) -> str:
    """소스 CSV 경로. HF 캐시에 받아 두고 재사용(오프라인이면 캐시 사용)."""
    from huggingface_hub import hf_hub_download

    e = KR2_EDITIONS[edition]
    return hf_hub_download(KR2_REPO, e["file"], repo_type="dataset", revision=e["revision"])


def _is_label(t: str) -> bool:
    return t.startswith("<") and t.endswith(">")


def _drop_labels(desc: str) -> str:
    """'키워드:' 뒤 쉼표 목록에서 <분류 라벨> 토큰을 뺀다. 남는 별칭이 없으면 '키워드:' 부분을 지운다."""
    head, sep, kw = desc.rpartition("키워드:")
    if not sep:
        return desc
    kept = [t.strip() for t in kw.split(",") if t.strip() and not (t.strip().startswith("<") and t.strip().endswith(">"))]
    return f"{head}{sep} {', '.join(kept)}" if kept else head.rstrip()


def _ref_keywords(edition: str) -> dict[str, list[str]]:
    """기준판(11-04 형식)의 태그 → '키워드:' 뒤 토큰 원본 (<라벨> 포함)."""
    out: dict[str, list[str]] = {}
    with open(source_path(edition), encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            _, sep, kw = row[3].rpartition("키워드:")
            if sep:
                out[row[0].strip().replace(" ", "_")] = [t.strip() for t in kw.split(",") if t.strip()]
    return out


def _strip_legacy_labels(aliases: list[str], ref_tokens: list[str]) -> list[str]:
    """11-10판 별칭에서 기준판의 <라벨> 이름을 뺀다 (모듈 독스트링 참고. 벤치 tagemb/kr2.py 와 같은 규칙)."""
    labels = [t[1:-1].strip() for t in ref_tokens if _is_label(t)]
    ref = [t for t in ref_tokens if not _is_label(t)]
    out = list(aliases)
    for x in labels:
        if out.count(x) > ref.count(x):
            out.remove(x)
    if out and ref and ref[-1] == out[-1] + ".":
        out[-1] = ref[-1]
    return out


def _convert_new(desc: str, ref_tokens: list[str] | None) -> str:
    """11-10판 '… 정의. <키워드> a, b.' → '… 정의. 키워드: a, b' (라벨 제거). 별칭이 없으면 '키워드:' 부분 없이."""
    i = desc.find(NEW_KEYWORD_TAG)
    head = desc[:i].rstrip()
    tail = desc[i + len(NEW_KEYWORD_TAG):].strip()
    if tail.endswith("."):  # 형식이 붙인 끝 마침표 한 개
        tail = tail[:-1]
    aliases = [t.strip() for t in tail.split(",") if t.strip() and not _is_label(t.strip())]
    if ref_tokens is not None:
        aliases = _strip_legacy_labels(aliases, ref_tokens)
    return f"{head} 키워드: {', '.join(aliases)}" if aliases else head


def iter_rows(min_count: int = MIN_COUNT, edition: str = KR2_EDITION):
    """(name, category, post_count, description) — 소스 순서(post_count 내림차순) 그대로."""
    ref_ed = KR2_EDITIONS[edition]["label_ref"]
    ref = _ref_keywords(ref_ed) if ref_ed else {}
    seen = set()
    with open(source_path(edition), encoding="utf-8") as f:
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
            desc = row[3] if len(row) > 3 else ""
            if NEW_KEYWORD_TAG in desc:
                yield name, cat, count, _convert_new(desc, ref.get(name))
            else:
                yield name, cat, count, _drop_labels(desc)


def write_tags_csv(out: str, min_count: int = MIN_COUNT, edition: str = KR2_EDITION) -> int:
    tmp = out + ".tmp"
    n = 0
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "category", "post_count", "description"])
        for r in iter_rows(min_count, edition):
            w.writerow(r)
            n += 1
    os.replace(tmp, out)
    return n


if __name__ == "__main__":
    from .config import TAGS_CSV_PATH

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(TAGS_CSV_PATH))
    ap.add_argument("--min-count", type=int, default=MIN_COUNT)
    ap.add_argument("--edition", default=KR2_EDITION, choices=sorted(KR2_EDITIONS))
    a = ap.parse_args()
    print(f"{write_tags_csv(a.out, a.min_count, a.edition)}개 태그 ({a.edition}) → {a.out}")
