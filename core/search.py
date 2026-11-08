"""
검색 모듈 (순수 벡터검색 빌딩블록)

LLM 없이 벡터 검색·어휘 일치·실존 태그 확인만 담당한다. variant(인덱스 이름, 현재 sim)를
파라미터로 받는다.

설계 의도:
    - LLM 오케스트레이션은 pipeline_structured.py(기본 모드)·pipeline_split.py(고급 모드)가 담당.
    - 이 모듈은 단독으로 import 하여 데이터 탐색/벤치마크에 바로 쓸 수 있다.
    - DB 인스턴스는 variant별로 캐싱하여 재연결 비용을 줄인다.
"""

import logging
import re
from dataclasses import dataclass

from .config import DEFAULT_TOP_K, CATEGORY_LABELS
from .database import TagDatabase, SearchResult, distance_to_similarity
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

# variant별 '태그 → 메타데이터' 캐시 (단어형+ 의 중복회피/생성보강용)
#   {tag: {"definition","aliases","major","minor"}}. tagset 과 동일한 lazy 패턴.
_definitions_cache: dict[str, dict[str, dict]] = {}


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


def get_definitions(variant: str) -> dict[str, dict]:
    """
    variant DB의 '태그 → 한국어 메타데이터' 매핑. (1회 로드 후 캐시)

    단어형+ 가 최종 태그의 의미 범위(definition/aliases/대·소분류)를 알아야
    "이미 태그가 커버한 부분"을 단어형에서 제외(또는 보강)할 수 있다.
    임베딩/벡터검색을 타지 않는 순수 테이블 덤프이며, get_tagset 과 동일하게
    첫 호출 1회만 LanceDB 를 읽고 이후엔 메모리 dict 조회다(단일워커+큐로 직렬화).

    반환: {tag: {"definition": str, "aliases": list[str], "major": str, "minor": str}}
    """
    if variant not in _definitions_cache:
        tbl = get_db(variant).table
        df = tbl.to_pandas()
        mapping: dict[str, dict] = {}
        for row in df.itertuples(index=False):
            # aliases 는 list 컬럼이라 to_pandas → itertuples 에서 numpy ndarray 로
            # 올 수 있다. `arr or []` 는 다중원소 array 의 truth value 가 모호해 터지므로
            # (ValueError: ambiguous), or 를 쓰지 말고 None 만 분기해 list() 로 감싼다.
            _al = getattr(row, "aliases", None)
            aliases = list(_al) if _al is not None else []
            mapping[row.tag] = {
                "definition": getattr(row, "definition", "") or "",
                "aliases": aliases,
                "major": getattr(row, "major", "") or "",
                "minor": getattr(row, "minor", "") or "",
            }
        _definitions_cache[variant] = mapping
        logger.info(f"definition 맵 로드 (variant={variant}): {len(mapping)}개")
    return _definitions_cache[variant]


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


# ---------------------------------------------------------------------------
# 이름 오염 차단 헬퍼 (split·일반 모드 공용)
# ---------------------------------------------------------------------------
# 캐릭터 이름이 cat0 일반검색을 타면 'name_(cosplay)' 같은 캐릭터 파생 태그를
# 잡는 오염이 생긴다. 캐릭터 정체성은 cat3/cat4 전용 검색이 담당하므로, 일반
# 검색 텍스트에서는 이름을 빼고 동작/속성/구도만 찾게 한다.

def strip_names_from_text(text: str, names: list[str]) -> str:
    """검색용 텍스트에서 인물/캐릭터 이름을 제거.

    표기 변형(띄어쓰기) 대응: "라이덴 쇼군"으로 등록해도 "라이덴쇼군"으로 붙여 쓴
    경우를 잡기 위해 이름 내부 공백을 '공백 0개 이상' 패턴으로 바꿔 매칭한다.
    긴 이름부터 제거(부분문자열 충돌 방지), 대소문자 무시.
    """
    out = text
    for nm in sorted([n.strip() for n in names if n.strip()], key=len, reverse=True):
        pat = r"\s*".join(re.escape(part) for part in nm.split())
        out = re.sub(pat, " ", out, flags=re.IGNORECASE)
    return out


_COSPLAY_RE = re.compile(r"\(cosplay\)\s*$", re.IGNORECASE)


def drop_character_derived(tags: list[str]) -> list[str]:
    """결과에서 _(cosplay) 등 캐릭터 파생 태그를 제거(이중 방어)."""
    return [t for t in tags if not _COSPLAY_RE.search(t)]


# ---------------------------------------------------------------------------
# structured 파이프라인 전용 검색 (기존 함수와 별개 — 기존 경로 동작 불변)
# ---------------------------------------------------------------------------
# 1) 카테고리를 LanceDB where 사전 필터로 거른다.
#    기존 TagDatabase.search 는 top_k*4 를 가져와 파이썬에서 거르는 사후 필터라, 상위권을 다른
#    카테고리가 채우면 결과가 비는 경우가 있다(벤치 실측: "여기사" → 로마자 한국 작가명(cat1)이 top20
#    을 채워 cat0 결과 0건).
# 2) 어휘 일치: 영어 조각을 태그 표기로 바꿔 같은 이름의 태그가 DB 에 있으면 후보로 쓴다.
#    (벤치 실측: nt 인덱스에서 "glasses" 질의 top5 에 glasses 가 없었다.)

# variant별 '태그 → (카테고리, 정의, 별칭)' 캐시 (어휘 일치용). 벡터 컬럼은 읽지 않는다.
_tagrow_cache: dict[str, dict[str, dict]] = {}


def get_tag_rows(variant: str) -> dict[str, dict]:
    """{tag: {"tag","category","definition","aliases"}} (1회 로드 후 캐시)."""
    if variant not in _tagrow_cache:
        tbl = get_db(variant).table
        cols = ["tag", "category", "definition", "aliases"]
        try:
            d = tbl.to_lance().to_table(columns=cols).to_pydict()
        except Exception:  # pylance 없는 환경 폴백
            df = tbl.to_pandas()
            d = {c: df[c].tolist() for c in cols}
        _tagrow_cache[variant] = {
            t: {"tag": t, "category": int(c), "definition": dfn or "",
                "aliases": list(al) if al is not None else []}
            for t, c, dfn, al in zip(d["tag"], d["category"], d["definition"], d["aliases"])
        }
        logger.info(f"태그 행 맵 로드 (variant={variant}): {len(_tagrow_cache[variant])}개")
    return _tagrow_cache[variant]


def lexical_lookup(en: str, variant: str, categories: set[int]) -> list[dict]:
    """영어 조각과 이름이 같은 태그(소문자·공백→_·단수/복수·관사 제거 변형). 없으면 []."""
    rows = get_tag_rows(variant)
    base = en.strip().lower().replace(" ", "_")
    forms = [base, base.rstrip("s") if base.endswith("s") else base + "s"]
    for art in ("a_", "an_", "the_"):
        if base.startswith(art):
            forms.append(base[len(art):])
    out = []
    for f in dict.fromkeys(forms):
        r = rows.get(f)
        if r and r["category"] in categories:
            out.append({**r, "score": None})
    return out


# ---------------------------------------------------------------------------
# 이름 정규화 (2026-11-08 운영 로그 표본 e10)
# ---------------------------------------------------------------------------
# 이름 비교용 정규화 (고급 모드에서 캐릭터 이름과 같은 속성 토큰·조각을 거를 때 쓴다)
_QUAL_RE = re.compile(r"(_\([^)]*\))+$")


def norm_name(s: str) -> str:
    """이름 비교용 정규화: 소문자, 글자·숫자만(공백·밑줄·구두점 제거)."""
    return re.sub(r"[\W_]+", "", (s or "").lower())


def tag_base(tag: str) -> str:
    """태그 본체: 끝의 괄호 한정어를 뗀 이름. laplace_(nikke) → laplace, saber_alter_(ver._shinjuku_1999) → saber_alter."""
    return _QUAL_RE.sub("", tag)


def search_prefiltered_multi(texts: list[str], categories: list[set[int]], variant: str, top_k: int = 5) -> list[list[dict]]:
    """
    여러 텍스트를 한 번에 임베딩하고, 텍스트마다 자기 카테고리로 where 사전 필터 검색 (structured 속도판).
    search_prefiltered 와 결과 형식이 같다. categories[i] 는 texts[i] 의 카테고리 집합.
    """
    if not texts:
        return []
    vecs = EmbeddingManager.embed_queries(texts)
    tbl = get_db(variant).table
    out: list[list[dict]] = []
    for v, cats in zip(vecs, categories):
        where = f"category IN ({', '.join(str(c) for c in sorted(cats))})"
        rows = tbl.search(v).where(where, prefilter=True).limit(top_k).to_list()
        out.append([{
            "tag": r["tag"], "category": r["category"], "definition": r.get("definition") or "",
            "aliases": list(r.get("aliases") or []),
            "score": round(distance_to_similarity(r["_distance"]), 4),
        } for r in rows])
    return out
