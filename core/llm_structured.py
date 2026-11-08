"""
structured 파이프라인의 LLM 프롬프트 (구조화 분해 / 조각별 번호 선택). 2026-11-05 속도판(벤치 v4)으로 갱신.
(이전 커밋에서는 파일명이 llm_v3.py 였다.)

프롬프트는 tag-embedding 벤치 gen_test/gemma_v4.py 에서 옮겼고, 2026-11-08 에 두 가지를 바꿨다.
  - 스키마 문자열 패턴(TEXT_PATTERN, 아래 설명)
  - 인원수 규칙 한 문장: "Animals are never counted, not even as others."
    새 모델(QAT Q4_K_M)이 동물을 others 로 세던 문제. 캐시 끔 리허설(로컬 LLM):
      실행 전에 고정한 새 문장 24개(홀드아웃) 한국어 16→21/24, 영어 17→23/24, 기존 40문장 37→39, 38→39.
      관계어 문장까지 더 길게 쓴 문구는 인원수가 1문항 더 맞았지만 "벽에 기댐" → [벽][기댐]처럼
      동사+목적어 조각을 쪼개는 부작용이 생겨(20문장 중 5문장 분해 변화) 이 짧은 문구를 골랐다(3문장 변화, 해롭지 않음).
    단어 목록은 넣지 않았다(과적합 방지).
판단 규칙은 v3 와 같고, 바뀐 것은 출력 형식(압축)과 인원수 규칙이다.
  - 분해 출력: {"girls","boys","others","characters":[[ko,en]],"units":[[ko,en]]}  (v3 160토큰 → 약 80토큰)
  - 선택 출력: {"u0":[0],"u1":"phrase"}  (v3 225토큰 → 약 34토큰)
  - 인원수: 동물은 사람 아님 / 이름·관계어로 성별 추론 / others 는 성별을 알 수 없을 때만
벤치(gemma4 26B A4B IQ4_XS) 10문장 개념 충족 94~95% (v3 95%), 분해+검색+선택 약 13초 (v3 약 30초).
예시는 벤치 시험 문장에 없는 개념으로만 만들었다(정답 유출 방지). 예시를 바꿀 때도 이 원칙을 지킬 것.
"""

# ── 1. 구조화 분해 ──
SYSTEM_DECOMPOSE = """You turn a Korean anime-image description into structured search units for a Danbooru tag database.

Output:
- girls / boys / others: how many people are in the image. Count only humans and human-like characters.
  Animals, objects, pictures and body parts are never people.
  girls = any female (소녀, 여자, 여성, 마녀, 여경찰 ...), boys = any male (소년, 남자, 남성 ...).
  Infer the gender from character names and relation words: "신랑과 신부" -> 1 boy, 1 girl; "삼촌과 조카딸" -> 1 boy, 1 girl;
  "카구야와 시로가네" -> 1 girl, 1 boy; "말 세 마리" -> nobody. Animals are never counted, not even as others. Use others only when the gender truly cannot be known.
  Named characters also count as people.
- characters: named existing characters or series as [Korean, English] (e.g. 프리렌, 라이덴 쇼군, 원신). Generic roles like 메이드, 간호사, 마법사 are NOT characters.
- units: every other visual element as [Korean, English], ONE concept per unit: hair color, hairstyle, eyes, accessory, clothing item, role/job, body feature, expression, pose, action, held object, place, background element, weather, time of day, lighting.

RULES
1. Person words never carry attributes. Split the attribute off into its own unit and put only the person into people.
   "금발 메이드 소녀" -> people [1 girl], units [금발], [메이드]
   "수염 난 남자" -> people [1 boy], units [수염]
   "여경찰" -> people [1 girl], units [경찰]      (gendered role: the person goes to people, the role stays a unit)
   "포니테일 남자 세 명" -> people [3 boy], units [포니테일]
2. Split compound scenery into separate concepts.
   "안개 낀 바닷가 절벽" -> [안개], [바닷가], [절벽]
   "비 내리는 숲속 오두막" -> [비], [숲], [오두막]
3. ONE action or state per unit. A verb stays together with its own object, but two different verbs are always two units, and a modifier that names a different visible thing becomes its own unit.
   "기타를 등에 메고" -> [기타를 등에 멤]                       (one verb with its object)
   "울면서 뛰어가는" -> [울음], [뛰어감]                          (two verbs -> two units)
   "의자에 기대어 하품하는" -> [의자에 기댐], [하품]
   "연기가 자욱한 술집" -> [연기], [술집]                        (the modifier names another thing)
   "눈보라 속에서 목도리를 여미며" -> [눈보라], [목도리를 여밈]
   "토끼 귀와 꼬리" -> [토끼 귀], [토끼 꼬리]
4. ko: the unit in short Korean as written in the sentence. en: a short lowercase English tag-like phrase for the SAME concept, translated with the whole sentence as context ("골목" -> "alley", "웅크리고" -> "squatting", "흑발" -> "black hair", "하품하며" -> "yawning").
5. Drop filler with no visual meaning (~하는 장면, ~인 모습, 있다).

Return ONLY compact JSON on a single line, matching the schema."""

# 문자열 안에 대괄호·큰따옴표를 못 쓰게 하는 패턴 (llama.cpp 가 스키마 문법으로 강제).
# 모델을 QAT Q4_K_M 으로 바꾼 뒤, 일부 문장에서 모델이 쌍의 첫 문자열 뒤에 배열을
# 닫으려다(스키마는 2개 항목을 요구) 닫는 기호를 문자열 안에 써 버려 ["눈], ","], ["] 처럼 깨졌다.
# 유효한 JSON 이라 파싱 실패로 잡히지 않고 조각이 사라졌다(새 장면 10개 중 2~3개, llama.cpp 두 빌드에서 모두 재현).
# 패턴을 걸면 모델이 문자열을 닫고 영어를 이어 쓴다. 정상 문장 17개는 출력이 글자 단위로 같았다(캐시 끔).
TEXT_PATTERN = r'^[^\[\]"]*$'
_TEXT = {"type": "string", "pattern": TEXT_PATTERN}
_PAIR = {"type": "array", "items": _TEXT, "minItems": 2, "maxItems": 2}
SCHEMA_DECOMPOSE = {
    "type": "object",
    "properties": {
        "girls": {"type": "integer", "minimum": 0, "maximum": 6},
        "boys": {"type": "integer", "minimum": 0, "maximum": 6},
        "others": {"type": "integer", "minimum": 0, "maximum": 6},
        "characters": {"type": "array", "items": _PAIR},
        "units": {"type": "array", "items": _PAIR},
    },
    "required": ["girls", "boys", "others", "characters", "units"],
    "additionalProperties": False,
}

# ── 2. 조각별 선택 (스키마는 호출 시 조각마다 만든다: pipeline_structured.select) ──
SYSTEM_SELECT = """You choose Danbooru tags for each search unit of an anime image description.

You get the original Korean sentence and, for each unit (u0, u1, ...), its Korean text, its English meaning, and numbered candidate tags from the database with their Korean definitions.

For EACH unit return either:
- a list of the numbers of the candidates that mean the SAME thing as the unit in this sentence (usually one, two only if both truly apply), or
- only when NO candidate matches, a short English phrase (2-5 words) for the unit, as a string.

A candidate matches only if it depicts the same thing. Sharing a word is not enough. Reject near-misses:
- unit "헬멧 / helmet" (worn): "helmet_removed" or "holding_helmet" is wrong; "helmet" is right.
- unit "귀걸이 / earrings": "single_earring" or "gold_earrings" is wrong when the sentence does not say so; "earrings" is right.
- unit "목도리를 여밈 / adjusting scarf": "scarf_over_mouth" or "scarf_removed" is wrong; "adjusting_scarf" is right.
- unit "팔짱 / crossed arms": "arm_hug" (hugging someone's arm) is wrong; "crossed_arms" is right.
- unit "골목 / alley": "alley_cat" is wrong; "alley" is right.
- unit "목걸이 / necklace": a specific variant like "pearl_necklace" is wrong when no material is stated; "necklace" is right.
Prefer the general tag that states exactly what the sentence says. Do not add colors, materials or details the sentence does not state.
Use a phrase only when no candidate matches; never use a phrase when a matching candidate exists.

Cover the whole unit. If a unit has two aspects (an action and an object, or a place and a position), pick one matching tag for each aspect when the candidates have them:
- unit "바이올린을 연주함 / playing violin": pick "violin" and "playing_instrument".
- unit "벽난로 앞 / in front of fireplace": pick "fireplace".
Do not leave a unit empty or use a phrase while a candidate expresses it.

Return ONLY compact JSON on a single line, like {"u0":[0],"u1":[2,4],"u2":"by the window"}."""
