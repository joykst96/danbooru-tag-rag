"""
파서 모듈 (이 프로젝트의 핵심 로직)

danbooru-tags.csv 의 description 컬럼은 일관성이 없다. 스캔 결과 4가지 타입:
    - ko_full  (62.4%): "[대분류 > 소분류] 정의문장. 키워드: 별칭1, 별칭2, ..."
    - empty    (21.7%): 빈 값
    - en_plain (14.2%): 영어 평문 설명
    - jp_plain ( 1.8%): 일본어 평문 설명

이 모듈은 각 행을 타입별로 분기 파싱하여 구조화된 ParsedTag 로 변환하고,
A/B 비교용 임베딩 텍스트를 생성한다.

핵심 결정(스캔 데이터 기반):
    - 별칭 파싱은 ko_full 71,142행에서 100% 성공 → 정규식 신뢰 가능
    - 한국어 정보가 없는 행(empty/en/jp, 약 38%)은 영어 태그명에 의존
      → 2-pass 검색의 '영어 pass'가 이 영역을 책임진다
"""

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 정규식 (스캔으로 검증됨)
# ---------------------------------------------------------------------------
# 맨 앞의 [ ... ] 메타 블록
_BRACKET_RE = re.compile(r'^\s*\[([^\]]+)\]\s*')
# "키워드:" 뒤의 쉼표구분 별칭
_KEYWORD_RE = re.compile(r'키워드\s*:\s*(.+?)\s*$')
# 한글 / 일본어 가나 존재 여부
_HANGUL_RE = re.compile(r'[가-힣]')
_KANA_RE = re.compile(r'[ぁ-んァ-ヶ]')


@dataclass
class ParsedTag:
    """파싱된 태그 1건."""
    tag: str                      # 영문 태그명 (예: "blue_hair")
    category: int                 # 0=일반,1=작가,3=작품,4=캐릭터,5=메타
    frequency: int                # post_count
    desc_type: str                # ko_full / en_plain / jp_plain / empty
    major: str | None = None      # 한국어 대분류 (예: "머리카락")
    minor: str | None = None      # 한국어 소분류 (예: "머리 색상")
    definition: str = ""          # 정의 본문 (메타/키워드 제거 후)
    aliases: list[str] = field(default_factory=list)  # 한국어 별칭

    @property
    def tag_readable(self) -> str:
        """언더스코어를 공백으로 바꾼 읽기용 태그명 (임베딩/표시에 사용)."""
        return self.tag.replace("_", " ")


def classify_description(desc: str) -> str:
    """description 을 4가지 타입 중 하나로 분류."""
    d = desc.strip()
    if not d:
        return "empty"
    has_bracket = bool(_BRACKET_RE.match(d))
    has_keyword = "키워드:" in d
    if has_bracket or has_keyword or _HANGUL_RE.search(d):
        if has_bracket and has_keyword:
            return "ko_full"
        return "ko_partial"  # 드묾(스캔상 39건). 한국어지만 정형이 일부 빠진 경우
    if _KANA_RE.search(d):
        return "jp_plain"
    return "en_plain"


def _split_hierarchy(bracket_content: str) -> tuple[str | None, str | None]:
    """'대분류 > 소분류' → (대분류, 소분류). 소분류 없으면 (대분류, None)."""
    parts = [p.strip() for p in bracket_content.split('>')]
    if not parts or not parts[0]:
        return None, None
    if len(parts) >= 2:
        return parts[0], parts[1]
    return parts[0], None


def _parse_aliases(desc: str) -> list[str]:
    """'키워드:' 뒷부분에서 한국어 별칭 리스트 추출."""
    m = _KEYWORD_RE.search(desc.strip())
    if not m:
        return []
    return [a.strip() for a in m.group(1).split(',') if a.strip()]


def _extract_definition(desc: str) -> str:
    """
    정의 본문만 추출.
    - 맨 앞 [대>소] 메타 블록 제거
    - 끝의 '키워드: ...' 블록 제거
    남은 가운데 설명문이 정의.
    """
    d = desc.strip()
    d = _BRACKET_RE.sub('', d)            # 앞 메타 제거
    d = _KEYWORD_RE.sub('', d).strip()    # 뒤 키워드 제거
    return d


def parse_row(tag: str, category: int, frequency: int, description: str) -> ParsedTag:
    """CSV 한 행을 ParsedTag 로 파싱한다."""
    desc_type = classify_description(description)

    parsed = ParsedTag(
        tag=tag,
        category=category,
        frequency=frequency,
        desc_type=desc_type,
    )

    if desc_type in ("ko_full", "ko_partial"):
        d = description.strip()
        m = _BRACKET_RE.match(d)
        if m:
            parsed.major, parsed.minor = _split_hierarchy(m.group(1))
        parsed.aliases = _parse_aliases(d)
        parsed.definition = _extract_definition(d)
    elif desc_type in ("en_plain", "jp_plain"):
        # 한국어 정보 없음. 정의 본문은 그대로 두되(참고용), 별칭/계층은 없음.
        parsed.definition = description.strip()
    # empty: 아무것도 없음. 영어 태그명에만 의존.

    return parsed


# ---------------------------------------------------------------------------
# A/B/C 임베딩 텍스트 생성
# ---------------------------------------------------------------------------
# variant A: 영어 태그명 + 한국어 정의 + 별칭  (전부 다 넣기)
# variant B: 영어 태그명 + 한국어 별칭          (정의 제외)
# variant C: 한국어 정의 + 별칭                  (영어 태그명 제외, 순수 한국어)
#
# 한국어 정보가 없는 행(empty/en/jp)에서는:
#   A, B → 영어 태그명이 항상 포함되므로 자연스럽게 영어로 검색됨
#   C   → 정의/별칭이 없으면 영어 태그명으로 폴백 (빈 문자열 방지)
# ---------------------------------------------------------------------------

def build_embed_text(parsed: ParsedTag, variant: str) -> str:
    """variant('a'|'b'|'c')에 맞는 임베딩 대상 문자열을 만든다."""
    alias_str = " ".join(parsed.aliases)
    tag_str = parsed.tag_readable

    if variant == "a":
        # 영어 태그명 + 정의 + 별칭 (전부)
        parts = [tag_str]
        if parsed.definition:
            parts.append(parsed.definition)
        if alias_str:
            parts.append(alias_str)
        return " ".join(parts).strip()

    elif variant == "b":
        # 영어 태그명 + 별칭 (정의 제외)
        parts = [tag_str]
        if alias_str:
            parts.append(alias_str)
        return " ".join(parts).strip()

    elif variant == "c":
        # 한국어 정의 + 별칭 (영어 태그명 제외). 둘 다 없으면 영어 태그명 폴백.
        parts = []
        if parsed.definition:
            parts.append(parsed.definition)
        if alias_str:
            parts.append(alias_str)
        if not parts:
            parts.append(tag_str)
        return " ".join(parts).strip()

    raise ValueError(f"알 수 없는 variant: {variant}")


# ---------------------------------------------------------------------------
# 단독 실행 시 간단 자가 점검 (CSV 일부로 파싱 결과 눈으로 확인)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        ("1girl", 0, 7598073,
         "[인물 > 인원수] 여성 캐릭터 한 명이 등장하는 이미지. 키워드: 1녀, 여캐, 여자애, 미소녀, 소녀"),
        ("blue_hair", 0, 100000,
         "[머리카락 > 머리 색상] 파란색 머리카락. 키워드: 청발, 파란 머리, 블루 헤어"),
        ("some_en_tag", 0, 500, "A plain english description without korean."),
        ("empty_tag", 0, 80, ""),
    ]
    for tag, cat, freq, desc in samples:
        p = parse_row(tag, cat, freq, desc)
        print(f"\n[{p.tag}] type={p.desc_type} cat={p.category}")
        print(f"  계층: {p.major} > {p.minor}")
        print(f"  정의: {p.definition}")
        print(f"  별칭: {p.aliases}")
        print(f"  embed_A: {build_embed_text(p, 'a')}")
        print(f"  embed_B: {build_embed_text(p, 'b')}")
        print(f"  embed_C: {build_embed_text(p, 'c')}")
