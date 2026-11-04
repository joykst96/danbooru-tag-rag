"""
v3 파이프라인의 LLM 프롬프트 (구조화 분해 / 조각별 번호 선택).

tag-embedding 벤치 gen_test/gemma_v3.py 의 프롬프트와 글자 단위로 같다(그대로 옮김).
벤치에서 gemma4 26B A4B(IQ4_XS)로 10문장 개념 충족률 95%/95% (Claude 95%/98%), 두 번 실행 10/10 동일.
예시는 모두 벤치 시험 문장에 없는 개념으로만 만들었다(정답 유출 방지). 예시를 바꿀 때도 이 원칙을 지킬 것.

v1(기존 4회 분해) 대비 바뀐 점과 근거는 pipeline_v3.py 머리말 참고.
"""

# ── 1. 구조화 분해: {people, characters, units[{ko, en}]} ──
SYSTEM_DECOMPOSE = """You turn a Korean anime-image description into structured search units for a Danbooru tag database.

Output three lists:
- people: who is in the image, as counts. type is "girl" (any female: 소녀, 여자, 여성, 마녀, 여경찰 ...), "boy" (any male: 소년, 남자, 남성 ...) or "other". Named characters also count as people.
- characters: named existing characters or series (e.g. 프리렌, 라이덴 쇼군, 원신). Generic roles like 메이드, 간호사, 마법사 are NOT characters.
- units: every other visual element, ONE concept per unit: hair color, hairstyle, eyes, accessory, clothing item, role/job, body feature, expression, pose, action, held object, place, background element, weather, time of day, lighting.

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

Return ONLY JSON matching the schema."""

SCHEMA_DECOMPOSE = {'type': 'object',
 'properties': {'people': {'type': 'array',
                           'items': {'type': 'object',
                                     'properties': {'count': {'type': 'integer', 'minimum': 1, 'maximum': 6},
                                                    'type': {'type': 'string',
                                                             'enum': ['girl', 'boy', 'other']}},
                                     'required': ['count', 'type'],
                                     'additionalProperties': False}},
                'characters': {'type': 'array',
                               'items': {'type': 'object',
                                         'properties': {'ko': {'type': 'string'}, 'en': {'type': 'string'}},
                                         'required': ['ko', 'en'],
                                         'additionalProperties': False}},
                'units': {'type': 'array',
                          'items': {'type': 'object',
                                    'properties': {'ko': {'type': 'string'}, 'en': {'type': 'string'}},
                                    'required': ['ko', 'en'],
                                    'additionalProperties': False}}},
 'required': ['people', 'characters', 'units'],
 'additionalProperties': False}

# ── 2. 조각별 선택: 후보 번호(enum)만 고를 수 있는 스키마는 호출 시 조각마다 만든다 (pipeline_v3._select) ──
SYSTEM_SELECT = """You choose Danbooru tags for each search unit of an anime image description.

You get the original Korean sentence and, for each unit (u0, u1, ...), its Korean text, its English meaning, and numbered candidate tags from the database with their Korean definitions.

For EACH unit return:
- pick: the numbers of the candidates that mean the SAME thing as the unit in this sentence. Usually one. Two only if both truly apply.
- phrase: null when pick is not empty. When NO candidate matches, pick = [] and phrase = a short English phrase (2-5 words) for the unit.

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

Return ONLY JSON matching the schema."""
