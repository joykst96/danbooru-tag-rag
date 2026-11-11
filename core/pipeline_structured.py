"""
structured 파이프라인 (기본 모드 본선. 고급 모드의 묘사·배경 칸도 describe() 로 같은 흐름을 쓴다)
(벤치 실험 이름: v3 → 속도판 v4. 이전 커밋에서는 파일명이 pipeline_v3.py 였다.)

tag-embedding 벤치(gen_test/gemma_v3.py → 속도판 gemma_v4.py)에서 gemma4 26B A4B 로 최적화·측정한 흐름을 옮겼다.
벤치 10문장 개념 충족률: 기존 4회 분해(v1) 82% → v3 95% → 속도판 94~95% (같은 조건 Claude 95~98%).
속도판(2026-11-05): 분해+검색+선택 약 30초 → 약 13초. 프롬프트 캐시 on, 출력 JSON 압축, 질의 임베딩 묶음.
  같은 서버 실측: 선택 출력 225 → 34토큰, 분해 출력 160 → 약 80토큰, 시스템 프롬프트 재처리(3~5초) 제거.

흐름 (LLM 호출: 분해 1 + 선택 1 + 자연어 1):
  1. 구조화 분해 (JSON 스키마 강제)
       girls/boys/others : 인원 → 코드가 1girl/2girls/1boy... 로 변환 (인원수를 검색·LLM 선별에 맡기지 않음)
                           동물은 사람 아님, 이름·관계어로 성별 추론, others 는 성별을 알 수 없을 때만
       characters        : 고유 캐릭터·작품 이름 [한국어, 영어]
       units             : 나머지 시각 개념, 조각 하나에 개념 하나 [한국어, 영어]
     기존 v1 의 실패: "단발머리 소년" 처럼 인물에 속성이 붙은 조각이 인원수 정규식에 걸려 통째로
     검색에서 빠짐(short_hair, glasses, miko, spacesuit 손실) → 인물·속성 분리 규칙으로 해결.
  2. 조각별 후보 = 어휘 일치(영어 표기와 같은 이름의 태그) + 한국어 질의 top5 + 영어 질의 top5
     (카테고리는 where 사전 필터. 일반 조각 cat0(또는 사용자가 고른 검색 풀), 캐릭터 조각 cat3/4)
  3. 선택: 조각마다 후보 "번호" 배열 또는 영어 구 문자열만 허용하는 JSON 스키마 → 후보 밖 태그는 구조적으로 불가.
     맞는 후보가 없으면 짧은 영어 구, 그것도 없으면 분해 때 받은 영어(en)를 구로 쓴다(조각을 버리지 않음).
  4. 최종 = 인원수 태그 + 캐릭터 태그(cat4) + 조각별 선택(태그·구).
     작품 태그(cat3)는 기존 정책대로 최종에서 뺀다. DB 실존 확인·_(cosplay) 제거는 기존과 같은 이중 방어.
  5. 자연어: 톤 기능(rich/plain/단어형+) (generate_nl_or_phrase).

스트리밍 이벤트(stage: korean / english / final / nl)는 이전 4회 분해 파이프라인(삭제됨)과 같은 모양이라
api.py·index.html 은 그대로 동작한다. final_tags 에는 태그 외에 영어 구가 섞일 수 있다(쉼표 제거).
"""

import logging
import re

from . import llm
from . import search as search_mod
from .config import MAIN_VARIANT, LLM_SEED
from .exceptions import LLMError
from .llm import _call_json
from .llm_structured import SYSTEM_DECOMPOSE, SCHEMA_DECOMPOSE, SYSTEM_SELECT, TEXT_PATTERN

logger = logging.getLogger(__name__)

TOP_K = 5                 # 질의(한국어·영어)마다 top5 → 조각당 후보 최대 10 + 어휘 일치
DEF_CHARS = 90            # 선택 프롬프트에 보여 줄 정의 길이
ALIAS_COUNT = 4           # 선택 프롬프트에 보여 줄 별칭 수
# 긴 입력 (2026-11-12 운영 로그 점검)
#   분해: 출력 한도 512 로는 태그 63개를 나열한 입력(803자)이 finish=length 로 매번 실패했다(오류 19건 전부).
#         입력이 길면 한도를 올린다. 2048 이면 같은 입력이 670토큰·조각 62개로 끝난다. 짧은 입력은 512 그대로다.
#   선택: 조각이 30개쯤 넘으면 프롬프트가 슬롯 컨텍스트(8192)를 넘어 400 이 나고 모든 조각이 분해 영어로 대체됐다
#         (기본 모드 성공 요청의 7%). 한 번에 부른 호출이 실패하면 SELECT_CHUNK 개씩 나눠 다시 부른다.
#         처음부터 나눠 부르지 않는 이유: 나눈 호출은 한 번에 부른 결과와 조각의 80~88%만 같았다(벤치 e14).
DECOMPOSE_LONG_CHARS = 300
DECOMPOSE_LONG_MAX_TOKENS = 2048
SELECT_CHUNK = 12


def count_tags(dec: dict) -> list[str]:
    """분해의 girls/boys/others → Danbooru 인원수 태그 (1girl / 2girls ... 6+girls, boy·other 동일)."""
    out = []
    for key, sing, plur in (("girls", "girl", "girls"), ("boys", "boy", "boys"), ("others", "other", "others")):
        c = int(dec.get(key, 0) or 0)
        if c == 1:
            out.append(f"1{sing}")
        elif 2 <= c <= 5:
            out.append(f"{c}{plur}")
        elif c >= 6:
            out.append(f"6+{plur}")
    return out


_BROKEN = set('[]"')
# 구(phrase) 문자열: 분해와 같은 문자 제한 + 빈 문자열 금지(패턴이 있으면 llama.cpp 는 minLength 를 보지 않는다)
PHRASE_PATTERN = TEXT_PATTERN.replace("*$", "+$")


def to_units(pairs: list) -> list[dict]:
    """
    분해의 [한국어, 영어] 쌍 → {"ko","en"}. 형식이 어긋난 항목은 버린다.
    대괄호·큰따옴표가 든 문자열은 JSON 구분자가 문자열 안으로 샌 붕괴 출력이라 버린다
    (스키마 패턴이 막지만, 패턴을 지원하지 않는 서버에 대비한 이중 방어).
    """
    out = []
    for p in pairs or []:
        if isinstance(p, (list, tuple)) and len(p) == 2 and str(p[0]).strip():
            ko, en = str(p[0]).strip(), str(p[1]).strip()
            if _BROKEN.isdisjoint(ko + en):
                out.append({"ko": ko, "en": en})
    return out


# 사람 낱말만 있는 조각 (분해 규칙 1 "사람 낱말은 people 로만"을 모델이 어긴 경우의 후처리).
# 2026-11-08 운영 로그 표본(e10)에서 "여성"/"소녀" 조각이 선택에서 female_orc·hamster_girl 같은 근접 오답에 붙고
# 자연어까지 오염됐다(n=32·41·75). 인원은 이미 girls/boys/others 로 셌으므로, 그 성별이 1명 이상 세어진
# 경우에만 버린다(세어지지 않았으면 그대로 둔다 = 기존 동작). 속성이 붙은 조각("성숙한 여성", "여군")은 안 건드린다.
# 이 목록은 코드 후처리용이다. 분해 프롬프트에는 넣지 않는다(프롬프트 한 줄 변경이 다른 규칙의 분해를 흔든 전례).
_PERSON_EN = {
    "girls": {"girl", "girls", "woman", "women", "female", "females", "lady", "ladies"},
    "boys": {"boy", "boys", "man", "men", "male", "males", "guy", "guys", "gentleman", "gentlemen"},
    None: {"person", "people", "persons", "human", "humans", "someone", "figure", "figures"},
}
_PERSON_KO = {
    "girls": {"여자", "여성", "소녀", "여인", "아가씨", "여자아이", "여자애"},
    "boys": {"남자", "남성", "소년", "남자아이", "남자애", "청년", "사내"},
    None: {"사람", "인물", "누군가", "사람들", "인물들"},
}
_COUNT_WORD_EN = {"a", "an", "the", "one", "two", "three", "four", "five", "six", "single", "several", "many"}
_COUNT_WORD_KO = re.compile(r"^(\d+명?|명|한|두|세|네|다섯|여섯|하나|둘|셋|넷|한명|두명|세명|네명)$")


def _person_gender(u: dict) -> str | None | bool:
    """조각이 사람 낱말만이면 그 성별 키("girls"/"boys", 성별 없는 낱말은 None), 아니면 False."""
    en = " ".join(w for w in u["en"].lower().replace("_", " ").split() if w not in _COUNT_WORD_EN)
    ko = "".join(w for w in u["ko"].split() if not _COUNT_WORD_KO.match(w))
    for g in ("girls", "boys", None):
        if en in _PERSON_EN[g] or ko in _PERSON_KO[g]:
            return g
    return False


def drop_counted_people(units: list[dict], dec: dict) -> tuple[list[dict], list[dict]]:
    """인원수로 이미 센 사람 낱말만 있는 조각을 뺀다. 반환: (남은 조각, 뺀 조각)."""
    people = {k: int(dec.get(k, 0) or 0) for k in ("girls", "boys", "others")}
    kept, dropped = [], []
    for u in units:
        g = _person_gender(u)
        counted = g is not False and (people[g] > 0 if g else sum(people.values()) > 0)
        (dropped if counted else kept).append(u)
    return kept, dropped


# 수식어만 남은 조각 합치기 (2026-11-09, 벤치 gen_test/e13_precision).
# 분해가 "검은색 밀리터리 복장"을 [검은색][밀리터리 복장]처럼 색·크기·재질 낱말을 따로 떼면, 그 조각이 선택에서
# 한 단어 구(black, wide)나 근접 오답(white_bird, purple_light, round_table)으로 남았다(e10 n=36·63·71).
# 분해 프롬프트를 고치는 길은 e10 94건 중 45건을 흔들고 회수를 떨어뜨려 보류했고(벤치 held/README.md),
# 대신 코드에서 '수식어 낱말만 있는 조각'을 바로 뒤 조각에 합쳐 한 번에 검색·선택한다. 해당 조각이 있는 문장만 바뀐다.
#   - 감정 낱말은 합치지 않는다("즐거움"처럼 따로 서는 사람 속성 조각).
#   - 뒤 조각이 머리·눈이면 합치지 않는다(분해 규칙 1: "갈색 단발머리" → brown_hair, short_hair 를 따로).
#   - 수식어 조각이 마지막이면 그대로 둔다.
# 이 어휘는 코드 후처리용이다. 분해 프롬프트에는 넣지 않는다.
_MOD_COLOR = {"black", "white", "red", "blue", "green", "yellow", "purple", "pink", "orange", "brown", "grey", "gray",
              "silver", "gold", "golden", "navy", "beige", "violet", "crimson", "scarlet", "indigo"}
_MOD_SHADE = {"light", "dark", "pale", "bright", "deep"}
_MOD_SIZE = {"big", "small", "large", "tiny", "huge", "giant", "wide", "narrow", "long", "short", "tall", "round"}
_MOD_MATERIAL = {"wooden", "wood", "metal", "metallic", "leather", "glass", "stone", "iron", "steel"}
_MOD_MANY = {"many", "numerous", "countless"}
_MOD_WORDS = _MOD_COLOR | _MOD_SIZE | _MOD_MATERIAL | _MOD_MANY
_MOD_FILLER = {"a", "an", "the", "color", "colored"}
_MOD_SKIP_NEXT = re.compile(r"\b(hair|eyes?|eyed|haired)\b")


def _modifier_only(en: str) -> bool:
    w = [x for x in re.split(r"[\s_\-]+", en.lower().strip()) if x and x not in _MOD_FILLER]
    return bool(w) and all(x in _MOD_WORDS or x in _MOD_SHADE for x in w) and any(x in _MOD_WORDS for x in w)


def merge_modifier_units(units: list[dict]) -> list[dict]:
    """영어 표기가 색·크기·재질 낱말뿐인 조각을 바로 뒤 조각과 합친다(위 설명). 합칠 것이 없으면 그대로."""
    out, i = [], 0
    while i < len(units):
        u = dict(units[i])
        while _modifier_only(u["en"]) and i + 1 < len(units) and not _MOD_SKIP_NEXT.search(units[i + 1]["en"].lower()):
            nxt = units[i + 1]
            u = {"ko": f"{u['ko']} {nxt['ko']}", "en": f"{u['en']} {nxt['en']}"}
            i += 1
        out.append(u)
        i += 1
    return out


# 인물 없는 장면 → no_humans (2026-11-08 운영 로그 표본 e10: "사람은 아무도 없고"(n=63), "인물이 없음"(n=64)이라고
# 써도 빠졌다. 분해가 이런 말을 '시각 의미 없는 말'로 버려 조각이 안 남는다. 옛 파이프라인(운영)은 붙였다.)
# 인원 0 이고 캐릭터가 없을 때만 본다. 둘 중 하나면 붙인다.
#   1) 입력에 사람이 없다는 말이 있다.
#   2) 최종 토큰이 전부 DB 태그(구가 없음)이고 서로 다른 태그가 2개 이상이며, 사람을 전제하는 대분류(신체·표정·의상·자세 ...)의
#      태그가 하나도 없고, 장면 대분류(배경·장소·자연 ...)의 태그가 하나 이상 있다 — 풍경·사물만 있는 장면
#      (Danbooru 관례상 no_humans). 구가 섞이면 판단하지 않는다("배구 서브하는 자세"처럼 구로만 남은 사람 동작을
#      놓치지 않으려고 보수적으로 잡음).
#      2026-11-12: 장면 태그가 있어야 한다는 조건과 사람 대분류 12개를 더했다. 대분류가 1,051종이라 사람 대분류
#      목록만으로는 못 막아서 "nude, full_body, from_behind"(성적·구성), "portrait, upper_body"(구도·구성),
#      "close-up, close-up"(같은 태그 둘)에도 no_humans 가 붙었다. 운영 로그에서 붙은 고유 입력 3건 중 2건이
#      이 오판정이었다(증명사진 얼굴, 초근접 접사). 평가 표본(e10)의 풍경 장면 6건은 그대로 붙는다.
_NO_PEOPLE_RE = re.compile(
    r"(사람|인물|인간)\S{0,2}\s*(아무도\s*)?없|아무도\s*없|\bno\s+(humans?|people|persons?|one|characters?)\b|\bnobody\b",
    re.IGNORECASE)
_PERSON_MAJORS = {"신체", "신체 특징", "표정", "패션", "의상", "복장", "의류", "자세", "포즈", "동작", "행동", "행위",
                  "인물", "외형", "머리", "머리카락", "장신구", "액세서리", "코스프레", "캐릭터", "종족", "성인", "H",
                  "성적 표현", "성행위",
                  "구성", "얼굴", "성적", "성", "상태", "표현", "신체 표현", "신체 장식", "의상 상태", "관계", "직업", "장식"}
_SCENE_MAJORS = {"배경", "장소", "자연", "풍경", "지리", "건축", "건물", "식물"}


def _majors(meta: dict) -> set[str]:
    """대분류 문자열 → 낱개 집합 ("배경/사물" 같은 겹친 표기를 나눈다)."""
    return {m.strip() for m in (meta.get("major") or "").split("/") if m.strip()}


def needs_no_humans(text: str, tokens: list[str], variant: str) -> bool:
    """인원 0·캐릭터 없음이 확인된 장면에서 no_humans 를 붙일지 (위 규칙). tokens = 최종 본문 토큰."""
    if _NO_PEOPLE_RE.search(text or ""):
        return True
    defs = search_mod.get_definitions(variant)
    tokens = list(dict.fromkeys(tokens))
    if len(tokens) < 2 or any(t not in defs for t in tokens):
        return False
    majors = [_majors(defs[t]) for t in tokens]
    return not any(m & _PERSON_MAJORS for m in majors) and any(m & _SCENE_MAJORS for m in majors)


# NovelAI 가중치 문법 정리 (2026-11-08 운영 로그 표본 e10 n=12: "2::장난스러운::", "-1.5:검은 머리::", "[[노란빛 머리]]").
# 그대로 두면 빼라는 뜻의 음수 가중치(-1.5:검은 머리::)가 black_hair 로 들어갔다.
# 가중치 구간 "w::text::"(또는 "w:text::")는 text 만 남기고, 음수 가중치 구간은 지운다. 겹친 괄호 [[ ]] {{ }} 는 벗긴다.
# "7:3 가르마"처럼 끝에 "::"가 없는 콜론은 건드리지 않는다. 문법이 없는 입력은 글자 그대로 돌려준다.
_WEIGHT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*::?\s*([^:,]*?)\s*::")


def strip_weight_syntax(text: str) -> str:
    if "::" not in text and not re.search(r"[\[\]{}]{2,}", text):
        return text
    t = _WEIGHT_RE.sub(lambda m: "" if float(m.group(1)) < 0 else m.group(2), text)
    t = re.sub(r"[\[\]{}]{2,}", "", t)
    t = re.sub(r"\s*,(\s*,)+", ",", t)
    return t.strip(" ,")


def all_candidates(units: list[dict], cats_list: list[set[int]], variant: str) -> list[list[dict]]:
    """
    조각별 후보: 어휘 일치 → 한국어 질의 → 영어 질의 순서로 합치고 중복 제거.
    모든 조각의 [한국어, 영어] 질의를 한 번에 임베딩한다(속도판, 조각마다 따로 하던 것을 묶음).
    """
    if not units:
        return []
    texts = [t for u in units for t in (u["ko"], u["en"])]
    cats = [c for c in cats_list for _ in (0, 1)]
    hits = search_mod.search_prefiltered_multi(texts, cats, variant, top_k=TOP_K)
    out = []
    for i, (u, c) in enumerate(zip(units, cats_list)):
        seen, merged = set(), []
        for h in search_mod.lexical_lookup(u["en"], variant, c) + hits[2 * i] + hits[2 * i + 1]:
            if h["tag"] not in seen:
                seen.add(h["tag"])
                merged.append(h)
        out.append(merged)
    return out


async def _select_call(korean_prompt: str, units: list[dict], cands: list[list[dict]], name: str) -> dict:
    """선택 LLM 호출 1회. 반환: {"u0": [번호...] 또는 "구", ...} (조각 번호는 넘겨받은 목록 기준)."""
    lines = [f"Original Korean sentence: {korean_prompt}", ""]
    props = {}
    for i, (u, cs) in enumerate(zip(units, cands)):
        lines.append(f'u{i}: "{u["ko"]}" / {u["en"]}')
        if not cs:
            lines.append("   (no candidates)")
        for j, h in enumerate(cs):
            al = ", ".join(h["aliases"][:ALIAS_COUNT])
            dfn = (h["definition"] or "")[:DEF_CHARS]
            lines.append(f'   {j}. {h["tag"]} — {dfn or "(no definition)"}' + (f" [aliases: {al}]" if al else ""))
        # 압축 출력: 조각마다 후보 번호 배열, 또는 맞는 후보가 없을 때만 영어 구 문자열
        nums = {"type": "array", "items": {"type": "integer", "enum": list(range(len(cs))) or [0]},
                "minItems": 1, "maxItems": 3}
        phrase = {"type": "string", "pattern": PHRASE_PATTERN}
        props[f"u{i}"] = {"anyOf": [nums, phrase]} if cs else phrase
    schema = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    return await _call_json(SYSTEM_SELECT, "\n".join(lines), schema, name, seed=LLM_SEED)


async def select(korean_prompt: str, units: list[dict], cands: list[list[dict]], name: str) -> tuple[list[list[str]], list[str]]:
    """
    조각별 선택. 반환: (조각별 선택 목록, 메모). 선택은 태그 또는 "phrase:<영어 구>".
    LLM 호출이 실패해도 조각을 버리지 않고 분해 때 받은 영어로 대체한다(메모에 남김).
    조각이 많아 한 번에 부른 호출이 실패하면 SELECT_CHUNK 개씩 나눠 다시 부른다(위 상수 설명).
    """
    notes: list[str] = []
    try:
        res = await _select_call(korean_prompt, units, cands, name)
    except LLMError as e:
        res = {}
        if len(units) > SELECT_CHUNK:
            logger.warning(f"선택 실패 → {SELECT_CHUNK}개씩 나눠 다시 부름 (조각 {len(units)}개): {e}")
            notes.append(f"선택 나눠 부름({len(units)}개): {e}")
            for start in range(0, len(units), SELECT_CHUNK):
                try:
                    part = await _select_call(korean_prompt, units[start:start + SELECT_CHUNK],
                                              cands[start:start + SELECT_CHUNK], name)
                except LLMError as e2:
                    logger.warning(f"선택 실패(나눈 호출 {start}~) → 분해 영어로 대체: {e2}")
                    notes.append(f"선택 실패: {e2}")
                    continue
                for k, v in part.items():
                    if k.startswith("u") and k[1:].isdigit():
                        res[f"u{start + int(k[1:])}"] = v
        else:
            logger.warning(f"선택 실패 → 분해 영어로 대체: {e}")
            notes.append(f"선택 실패: {e}")

    picks: list[list[str]] = []
    for i, (u, cs) in enumerate(zip(units, cands)):
        r = res.get(f"u{i}")
        if isinstance(r, str) and _NUM_STR.match(r):
            # 번호를 문자열로 낸 경우("4", "1, 0"): 범위 안이면 후보 번호로 해석, 아니면 버리고 분해 영어로
            nums = [int(x) for x in re.findall(r"\d+", r)]
            notes.append(f"{u['ko']}: 숫자 문자열 {r!r} → 번호로 해석")
            r = nums
        elif isinstance(r, str) and _INSTRUCTION_LEAK.search(r):
            notes.append(f"{u['ko']}: 지시문 문구 {r!r} → 분해 영어 '{u['en']}'")
            r = None
        elif isinstance(r, str) and r.strip():
            # 태그 이름을 문자열로 쓴 경우 (아래 _tag_key 설명)
            key = _tag_key(r)
            names = [h["tag"] for h in cs]
            if key in names:
                notes.append(f"{u['ko']}: 태그 이름 문자열 {r!r} → 후보 번호로 해석")
                r = [names.index(key)]
            elif "_" in r:
                notes.append(f"{u['ko']}: 후보에 없는 태그 표기 {r!r} → 분해 영어 '{u['en']}'")
                r = None
        if isinstance(r, list):
            tags = list(dict.fromkeys(cs[j]["tag"] for j in r if isinstance(j, int) and 0 <= j < len(cs)))
            picks.append(tags or ["phrase:" + u["en"]])
        elif isinstance(r, str) and r.strip():
            picks.append(["phrase:" + r.strip()])
        else:
            picks.append(["phrase:" + u["en"]])
            notes.append(f"{u['ko']}: 응답 없음 → 분해 영어 '{u['en']}'")
    return picks, notes


# 선택 응답의 구 문자열 검사 (2026-11-08 운영 로그 표본 e10).
# - 숫자만 있는 문자열: 스키마가 anyOf[번호 배열, 구 문자열]이라 모델이 "u0":"4" 처럼 번호를 문자열로 내면
#   구로 처리돼 최종 태그에 "4, 1, 0"이 찍혔다(n=73, n=17). 구 패턴에 영문자를 필수로 넣는 방법도 있지만
#   문법이 바뀌면 모델 출력 전체가 달라질 수 있어(번호 대신 엉뚱한 글자를 이어 쓰게 됨) 코드에서 해석한다.
# - 지시문 문구: 프롬프트의 "when NO candidate matches"가 구로 새어 나왔다(n=63 "no candidate matches").
# - 태그 이름 문자열 (2026-11-12 운영 로그 점검, 고유 요청 207건·조각 1,108개): 구 110개 중 46개는 그 조각의
#   후보 태그 이름을 번호 대신 글자로 쓴 것이었고("u3":"on_floor"), 8개는 DB 에 없는 밑줄 표기였다
#   ("censored_breasts", "college_student"). 앞의 것은 그 후보로 해석하고(화면 표시는 같지만
#   로그·구 빼기 옵션·no_humans 판정에서 태그로 다뤄진다), 뒤의 것은 태그처럼 보이는 가짜라 분해 영어로 바꾼다
#   — "후보 밖 태그는 내보내지 않는다"는 원칙을 문자열 경로에도 적용한다. 밑줄이 없는 보통 구는 그대로 둔다.
_NUM_STR = re.compile(r"^\s*\d+(\s*[,/ ]\s*\d+)*\s*$")


def _tag_key(s: str) -> str:
    """구 문자열 → 태그 표기(소문자, 공백을 밑줄로)."""
    return "_".join(s.strip().lower().split())
_INSTRUCTION_LEAK = re.compile(r"\bcandidates?\b|\bno\s+match|^\s*(none|n/?a|null|nothing)\s*$", re.IGNORECASE)


async def decompose(text: str) -> dict:
    """구조화 분해 1회 (JSON 스키마 강제). 입력이 길면 출력 한도를 올린다(DECOMPOSE_LONG_* 설명)."""
    if len(text) > DECOMPOSE_LONG_CHARS:
        return await _call_json(SYSTEM_DECOMPOSE, text, SCHEMA_DECOMPOSE, "decompose", seed=LLM_SEED,
                                max_tokens=DECOMPOSE_LONG_MAX_TOKENS)
    return await _call_json(SYSTEM_DECOMPOSE, text, SCHEMA_DECOMPOSE, "decompose", seed=LLM_SEED)


# 내용 없는 구 (2026-11-12 운영 로그 점검). 분해가 위치·연결 낱말을 따로 떼면("바이크 앞에" → [바이크][앞에])
# 그 조각은 맞는 태그가 없어 "in front of", "inside", "wearing", "covered", "expression" 같은 구로 최종 태그에
# 남았다(조각 1,108개 중 29개). 그 낱말만으로는 그릴 것이 없으므로 최종 태그에서 뺀다.
#   - 구로 남았을 때만 뺀다. 같은 부류라도 태그가 붙은 조각(from_behind, straight-on, looking_at_viewer)은 정상이다.
#   - 시점·방향 낱말(from, behind, above, below, side, back, view, shot, angle)은 넣지 않았다
#     ("from below" 처럼 구로 남아도 뜻이 있다).
#   - 분해 영어(en_units)는 그대로라, 담백 톤 자연어의 입력에는 남는다.
# 분해 프롬프트로 막는 길은 조각을 통째로 잃는 회귀가 있어 버렸다(벤치 gen_test/e14_audit).
# 이 어휘는 코드 후처리용이다. 프롬프트에는 넣지 않는다.
_EMPTY_WORDS = set("""
a an the in on at of to by with into onto inside within near beside between around across along through front
is are be being been was were has have having
wearing worn put placed located visible shown showing seen looking look
open opened closed covered covering fixed lit entering
thing something object part area place spot shape form
expression facial face appearance pose posture gesture motion movement action state situation scene
outfit clothes clothing attire background setting atmosphere mood feeling
random various some any one two both each every all only just very really
secretly gently slowly quickly slightly intense naturally
""".split())


def _empty_phrase(text: str) -> bool:
    """구가 위 낱말로만 이뤄졌는지."""
    w = [x for x in re.split(r"[\s_\-]+", text.lower().strip()) if x]
    return bool(w) and all(x in _EMPTY_WORDS for x in w)


def assemble(picks: list[list[str]], all_cands: list[list[dict]], n_chars: int,
             variant: str, empty_out: list[str] | None = None) -> tuple[list[str], list[str], list[str]]:
    """
    선택 결과 → (캐릭터 태그, 본문 토큰, DB 밖이라 지운 태그). 앞의 n_chars 개 조각이 캐릭터 조각.
    내용 없는 구(_EMPTY_WORDS)는 본문에 넣지 않고, empty_out 을 주면 거기에 적는다(로그용).
    캐릭터 조각은 cat4 태그만 남긴다(작품 cat3 은 최종에서 제외, 기존 정책).
    본문은 태그와 영어 구(쉼표 제거)가 섞이며, DB 실존 확인과 _(cosplay) 등 파생 태그 제거를 거친다.
    """
    tag_cat = {h["tag"]: h["category"] for cs in all_cands for h in cs}
    char_tags, body = [], []
    for i, ps in enumerate(picks):
        for p in ps:
            if p.startswith("phrase:"):
                t = _phrase_text(p)
                if t and i >= n_chars and _empty_phrase(t):
                    if empty_out is not None and t not in empty_out:
                        empty_out.append(t)
                elif t:
                    body.append(t)
            elif i < n_chars:
                if tag_cat.get(p) == 4 and p not in char_tags:
                    char_tags.append(p)
            else:
                body.append(p)
    db_tags = [t for t in body if t in tag_cat]
    _, dropped_db = search_mod.filter_existing_tags(db_tags, variant)
    dropped = set(dropped_db)
    body = [t for t in body if t not in dropped]
    body = search_mod.drop_character_derived(body)
    return char_tags, body, list(dropped_db)


async def describe(text: str, variant: str = MAIN_VARIANT, strip_names: list[str] | None = None) -> dict:
    """
    고급 모드의 묘사·배경 칸용: 분해 → 일반(cat0) 조각만 검색·선택 → 본문 토큰.
    분해의 characters 는 버린다(고급 모드는 캐릭터를 이름 칸으로 따로 찾는다).
    strip_names: 조각의 한국어 표기에서 이 이름들을 지운다(지우고 비면 조각을 버림) —
                 이름이 cat0 검색을 타면 'name_(cosplay)' 같은 파생 태그가 잡히기 때문.
                 영어 표기가 이름과 똑같은 조각도 버린다: 이름을 영어로 적은 칸("sakura miko")에서 분해가
                 한국어 표기("사쿠라 미코")로 조각을 만들면 한국어 지우기에 안 걸려 'sakura miko' 구가 남았다(e10 n=5).
    반환: {"tags": 본문 토큰, "hallucinated": DB 밖 태그, "people": {"girls","boys","others"},
           "characters": 분해의 characters [{"ko","en"}] (고급 모드가 영어 이름을 얻는 데 쓴다)}
    """
    empty = {"tags": [], "hallucinated": [], "people": {"girls": 0, "boys": 0, "others": 0}, "characters": []}
    text = strip_weight_syntax(text)
    if not text.strip():
        return empty
    dec = await decompose(text)
    name_keys = {search_mod.norm_name(n) for n in (strip_names or [])} - {""}
    units = []
    for u in merge_modifier_units(drop_counted_people(to_units(dec.get("units")), dec)[0]):
        ko = search_mod.strip_names_from_text(u["ko"], strip_names).strip() if strip_names else u["ko"]
        if ko and search_mod.norm_name(u["en"]) not in name_keys:
            units.append({"ko": " ".join(ko.split()), "en": u["en"]})
    people = {k: int(dec.get(k, 0) or 0) for k in ("girls", "boys", "others")}
    chars = to_units(dec.get("characters"))
    if not units:
        return {**empty, "people": people, "characters": chars}
    cands = all_candidates(units, [set(search_mod.GENERAL_CATS)] * len(units), variant)
    picks, notes = await select(text, units, cands, variant)
    if notes:
        logger.info(f"structured(묘사) 선택 메모: {notes}")
    _, body, dropped = assemble(picks, cands, 0, variant)
    return {"tags": list(dict.fromkeys(body)), "hallucinated": dropped, "people": people, "characters": chars}


async def generate_nl_or_phrase(
    korean_prompt: str,
    kept: list[str],
    en_units: list[str],
    nl_tone: str | None,
    variant: str,
    char_tags: list[str] | None = None,
    temperature: float | None = None,
) -> str:
    """
    자연어 단계 출력 분기: 일반 톤이면 자연어 문장, "phrase"면 단어형 보조출력.

    단어형(UI '단어형+')은 확정 태그(kept)가 표현 못 한 잔차만 짧은 영어 구로 낸다.
    중복 배제 기준이 kept 이므로 kept 를 그대로 넘기고, 태그의 의미 범위(definition)를
    배제맥락으로 함께 준다.

    톤별 입력(NL 의 목적은 'tag 수렴 과정에서 잘려나간 손실 복구'):
      - rich  : kept(최종 태그) 기반. 정합성 우선, 묘사적.
      - plain : en_units(DB 통과 전 영어 분해단위) 기반. kept 를 받으면 손실된 결과를
                받아 손실을 복구하라는 모순이 되므로, 수렴 전 정보를 입력으로 준다.
                캐릭터 정체성은 char_tags 를 앞에 보존해 SYSTEM_NL RULE2 의 'X from Y' 오프닝을 살린다.
                en_units 가 비면 kept 로 폴백.
    temperature: None 이면 톤별 권장 온도(llm.NL_TONES / PHRASE_TEMPERATURE). UI 슬라이더 값이 오면 그 값.
    """
    if nl_tone == "phrase":
        defs = search_mod.get_definitions(variant)
        return await llm.generate_phrase(korean_prompt, kept, defs=defs, temperature=temperature)
    if nl_tone == "plain":
        basis = en_units or kept
        if char_tags:
            basis = char_tags + [t for t in basis if t not in char_tags]
        return await llm.generate_nl_prompt(korean_prompt, basis, tone=nl_tone, temperature=temperature)
    # rich(및 기타): 최종 태그 기반. 태그가 비면 영어 분해단위로 폴백.
    nl_basis = kept or en_units
    return await llm.generate_nl_prompt(korean_prompt, nl_basis, tone=nl_tone, temperature=temperature)


def _phrase_text(p: str) -> str:
    """구 → 최종 프롬프트 토큰. UI 가 ', ' 로 태그를 나누므로 쉼표는 지운다."""
    return " ".join(p[len("phrase:"):].replace(",", " ").split())


def drop_phrases(tokens: list[str], variant: str) -> tuple[list[str], list[str]]:
    """
    DB 태그가 아닌 영어 구를 뺀다 (UI '영어 구 포함' 끔 = 요청 include_phrases=False). 반환: (남은 토큰, 뺀 구).
    선택이 구로 낸 것이라도 공백을 언더스코어로 바꿔 DB 태그와 같으면("looking at viewer") 태그로 보고 남긴다 —
    UI 는 어차피 언더스코어를 공백으로 바꿔 보여 주므로 출력 문자열이 같다.
    인원수·캐릭터 태그에는 쓰지 않는다(호출하는 쪽이 본문 토큰에만 적용).
    """
    tagset = search_mod.get_tagset(variant)
    kept, dropped = [], []
    for t in tokens:
        (kept if t in tagset or t.replace(" ", "_") in tagset else dropped).append(t)
    return kept, dropped


async def stream_structured_pipeline(
    korean_prompt: str,
    variant: str = MAIN_VARIANT,
    generate_nl: bool = True,
    search_categories: set[int] | None = None,
    nl_tone: str | None = None,
    nl_temperature: float | None = None,
    include_phrases: bool = True,
):
    """
    structured 를 단계별로 스트리밍. yield {"stage": korean|english|final|nl, "status": 표시 문구, "data": dict}.
    include_phrases=False: 최종 태그에서 DB 밖 영어 구를 뺀다(drop_phrases). no_humans 판정과 자연어 입력은
    구를 빼기 전 목록을 그대로 쓴다 — 태그 칸만 바뀌고 자연어 결과는 켰을 때와 같다.
    """
    unit_cats = set(search_categories) if search_categories else set(search_mod.GENERAL_CATS)
    korean_prompt = strip_weight_syntax(korean_prompt)

    # ── 1. 구조화 분해 ──
    dec = await decompose(korean_prompt)
    chars = to_units(dec.get("characters"))
    units, dropped_people = drop_counted_people(to_units(dec.get("units")), dec)
    if dropped_people:
        logger.info(f"structured 사람 낱말 조각 제외(인원수로 셈): {[u['ko'] + '/' + u['en'] for u in dropped_people]}")
    units = merge_modifier_units(units)
    count = count_tags(dec)
    logger.info(f"structured 분해: 인원={count} 캐릭터={[c['ko'] for c in chars]} 조각={[u['ko'] + '/' + u['en'] for u in units]}")
    yield {"stage": "korean", "status": "한국어 키워드 추출 완료",
           "data": {"ko_units": [c["ko"] for c in chars] + [u["ko"] for u in units], "person_units": count}}
    en_units = [c["en"] for c in chars] + [u["en"] for u in units]
    yield {"stage": "english", "status": "영어 키워드 추출 완료", "data": {"en_units": en_units}}

    # ── 2. 조각별 후보 ──
    all_units = chars + units
    all_cands = all_candidates(all_units, [set(search_mod.CHARACTER_CATS)] * len(chars) + [unit_cats] * len(units),
                               variant)

    # ── 3. 선택 ──
    picks, notes = await select(korean_prompt, all_units, all_cands, variant)
    if notes:
        logger.info(f"structured 선택 메모: {notes}")

    # ── 4. 최종 조립 + 이중 방어 ──
    empty_phrases: list[str] = []
    char_tags, body, dropped_db = assemble(picks, all_cands, len(chars), variant, empty_phrases)
    if not count and not chars and "no_humans" not in body and needs_no_humans(korean_prompt, body, variant):
        count = ["no_humans"]
    final = list(dict.fromkeys(count + char_tags + body))
    final_out, phrases_dropped = final, []
    if not include_phrases:
        body_kept, phrases_dropped = drop_phrases(body, variant)
        final_out = list(dict.fromkeys(count + char_tags + body_kept))
    # 생성 로그용 추적(api 가 파일 로그에만 남기고 클라이언트로는 보내지 않는다).
    # 조각마다 분해(ko/en) → 후보 태그 이름 → 선택을 한 줄로 이어서, 틀린 태그가 검색 탓인지 선택 탓인지 로그만 보고 가른다.
    trace = {
        "people": count,
        "dropped_people": [u["ko"] + "/" + u["en"] for u in dropped_people],
        "units": [
            {"kind": "character" if i < len(chars) else "unit", "ko": u["ko"], "en": u["en"],
             "cands": [h["tag"] for h in cs], "picks": p}
            for i, (u, cs, p) in enumerate(zip(all_units, all_cands, picks))
        ],
        "select_notes": notes,
    }
    if phrases_dropped:
        trace["phrases_dropped"] = phrases_dropped
    if empty_phrases:
        trace["empty_phrases"] = empty_phrases
    yield {"stage": "final", "status": "태그 완성",
           "data": {"final_tags": final_out, "hallucinated": list(dropped_db), "trace": trace}}

    # ── 5. 자연어 (기존 톤 기능 그대로. 구를 뺐어도 빼기 전 final 을 쓴다) ──
    nl_prompt = ""
    if generate_nl:
        nl_prompt = await generate_nl_or_phrase(korean_prompt, final, en_units, nl_tone, variant,
                                                char_tags=char_tags, temperature=nl_temperature)
    yield {"stage": "nl", "status": "완료", "data": {"nl_prompt": nl_prompt}}


async def run_structured_pipeline(korean_prompt: str, variant: str = MAIN_VARIANT, generate_nl: bool = False,
                          search_categories: set[int] | None = None, nl_tone: str | None = None,
                          include_phrases: bool = True) -> dict:
    """스트림을 모아 dict 로 반환 (검증·벤치마크용)."""
    out: dict = {}
    async for ev in stream_structured_pipeline(korean_prompt, variant, generate_nl, search_categories, nl_tone,
                                               include_phrases=include_phrases):
        out.update(ev["data"])
    return out
