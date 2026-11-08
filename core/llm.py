"""
LLM 모듈

로컬 LLM(llama.cpp 등 OpenAI 호환 엔드포인트) 호출 함수들.
구조화 호출(_call_json: 분해·선택)은 pipeline_structured, 자연어·단어형·캐릭터 선택은 두 파이프라인이 함께 쓴다.

모든 함수는 thinking off, 낮은 temperature(변별 작업) 전제로 설계되었으나
temperature 는 인자로 조절 가능하다. 자연어 생성만 약간 높은 temperature 권장.
"""

import re
import json
import logging

import httpx

from .config import (
    LLM_API_URL,
    LLM_MODEL,
    LLM_THINKING_ENABLED,
    LLM_CACHE_PROMPT,
)
from .exceptions import LLMError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 시스템 프롬프트
# ---------------------------------------------------------------------------
SYSTEM_NL = """You are a prompt writer for the ANIMA image generation model.
Given a Korean intent and English keywords, write a natural-language English prompt that conveys the keywords.
Rules:
1. Cover the given keywords. Do not add subjects, objects, or attributes that are not in the keywords or Korean intent.
2. CHARACTER OPENING: If the keywords contain a character tag, the prompt MUST start with that character, then describe appearance. Danbooru character tags look like "name_(series)" or just "name".
   - With series: start "<Name> from <Series>, ..." (e.g. tag "frieren_(sousou_no_frieren)" -> "Frieren from Sousou no Frieren, ..."). Convert underscores to spaces, use standard capitalization.
   - Without series (bare name tag): start "<Name>, ...".
   - The word "from" linking a character to its series must always be kept, regardless of tone.
   - If MULTIPLE character tags are present, open with each character (name first, then appearance), one per clause.
3. If NO character tag is present, use a plain noun like 'A girl', 'A boy', 'A man', 'A woman' to open, then appearance.
4. Standard capitalization for character/series names; convert tag underscores to spaces in the prose.
{TONE_RULES}
Return ONLY the raw prompt text. No markdown, no explanations."""

# ── 분할입력(고급) 전용: 캐릭터/작품 태그 선택 ──
# 사용자가 입력한 캐릭터명·작품명과, DB 검색으로 얻은 후보군을 함께 주고
# LLM이 가장 부합하는 캐릭터 태그와 작품 태그를 고르게 한다.
# (벡터 점수 top1 단순 채택은 "rover" 같은 동명이인/동일작품 내 조연 구분을 못 함.)
SYSTEM_SELECT_CHARACTER = """You select the correct Danbooru character tag and copyright(series) tag.

You are given:
- The user's intended CHARACTER name (as they typed it, possibly Korean or English).
- The user's intended SERIES name (optional).
- CHARACTER_CANDIDATES: Danbooru character tags retrieved from the DB, each with a score and aliases.
- SERIES_CANDIDATES: Danbooru copyright tags retrieved from the DB, each with a score and aliases.

Pick the single best character tag and the single best series tag from the candidates.

RULES:
1. Choose ONLY from the provided candidates. Never invent a tag not in the lists. If nothing fits, return empty string for that field.
2. The user's intent dominates the vector score. A lower-scored candidate that matches the user's stated name/series is correct over a higher-scored mismatch.
3. If a SERIES name is given, prefer the character candidate that belongs to that series. Danbooru character tags look like "name_(series)" — match the series inside the parentheses to the chosen series tag.
4. Disambiguate within the same series by the character NAME the user typed (e.g. user "프리렌"/"frieren" → pick "frieren", NOT a side character like "linie_(sousou_no_frieren)").
5. Match across languages and aliases (Korean name vs English tag vs alias).
6. If the user's character clearly is not among the candidates, return "" for character (do not force a wrong pick).

Return ONLY a JSON object, no markdown:
{"character": "<tag or empty>", "series": "<tag or empty>"}"""

# ── 분할입력(고급) 전용: 인물 단위 자연어 프롬프트 ──
# Anima 공식 허깅페이스 팁:
#   "Name a character, then describe their basic appearance."
#   다인물일수록 중요 — 이름만 나열하고 외형묘사가 없으면 모델이 인물을 헷갈린다.
# 따라서 각 인물마다 [이름(+작품)] + [외형/속성 묘사]를 반드시 한 덩어리로 붙여 쓴다.
# 캐릭터명이 DB에 없어 태그가 안 잡힌 경우에도 사용자가 적은 이름 문자열을 그대로
# 살려 쓰되, 그 칸의 묘사 태그로 외형을 채운다(이름만 덩그러니 두지 않는다).
SYSTEM_NL_MULTI = """You are an expert prompt engineer for the ANIMA anime image generation model.

You are given a SCENE made of one or more CHARACTER blocks and an optional BACKGROUND block. Write a single descriptive English prompt for the whole scene.

CORE RULE (critical for multi-character scenes):
For EACH character, name the character first, then immediately describe their appearance. Never list character names without appearance description — the model confuses characters otherwise. Even when a name is given, weave in the appearance tags for that character.

WHEN THERE ARE 2 OR MORE CHARACTERS:
Describe each character in its own SEPARATE sentence. Do NOT pack multiple characters into one sentence, and do NOT blend their attributes. One sentence per character, each starting with that character's name or handle (e.g. "<Name A> ... ." then "<Name B> ... ."). Only mention spatial position (left/right/center) if the description actually specifies it — do not invent positions. Keeping characters in distinct sentences is what prevents the model from mixing them up.

RULES:
1. Write each character as its own clause/sentence: "<Name> from <Series>, with <appearance woven from the tags>, ...". If a character has a series, include it once.
2. If a block has a name but NO matched tags, still use the name and describe whatever the block's description implies — do not drop the character. A block marked with the [new] flag is a new/unlisted character: use its given name as-is as the character handle even though no canon tags matched.
3. If a block is marked with the [original] flag, it has no canon name AND no series. Invent a simple fitting first name for that character and use it the same way (name first, then appearance). NEVER invent or attach a series/work to an original character, and NEVER use the word "from" for it — an original character is not "from" anything. Write just the invented name followed by the appearance (e.g. "Aria, with long silver hair, ..."). This gives the model a stable handle so it does not confuse this character with others.
4. If a block has NO name and is not original, use a plain noun ("A girl", "A boy", "A figure") consistent with the description, then the appearance.
5. Keep every character visually distinct so the model can separate them. Do not merge two characters' attributes into the same sentence.
6. End with the background/setting as a shared clause if a BACKGROUND block is given.
7. Use standard capitalization for character/series names. Do not add attributes that are not in the given tags/description.
8. SOURCE STRUCTURE (always preserved, regardless of tone): whenever a character actually has a series, keep the "<Character> from <Series>" structure intact (e.g. "Fern from Sousou no Frieren"). The linking word "from" must never be dropped or replaced, even under a compressed tone. Tone may shorten the appearance description, but not this name-to-series link. This rule applies ONLY to characters that genuinely have a series — it must NEVER be used to attach a "from <Series>" to an [original] character (originals have no series; see rule 3).
{TONE_RULES}
9. Return ONLY the raw prompt text. No markdown, no headings, no explanations.
10. NEVER copy the input's structural labels into the prompt. The input uses labels like "CHARACTER 1:", "BACKGROUND:", name=, series=, tags=, description=, original_character=true, "(new/unlisted character ...)", and bracket flags such as [original]/[new]. These are instructions to you, NOT words to write. The output must read as a natural scene description with no key=value pairs, no block headers, and no parenthetical meta-notes about a character being original or unlisted."""


# ---------------------------------------------------------------------------
# 단어형(phrase) 보조 출력 — 자연어 칸 대체용
# ---------------------------------------------------------------------------
# 완전한 문장이 길어질수록 ANIMA 그림체가 흔들리는 문제 때문에, 문장 대신 짧은
# 영어 구(phrase)를 콤마로 나열한다. 핵심: 이미 확정된 Danbooru 태그가 표현하지
# '못한' 잔차(시각 디테일/분위기)만 담는다. 태그와 의미가 겹치면 안 된다.
SYSTEM_PHRASE = """You write a SHORT comma-separated list of English phrases that COMPLEMENT a set of already-decided Danbooru tags for the ANIMA image model.

The Danbooru tags already cover part of the user's Korean intent. Your job is to capture ONLY what those tags do NOT express — extra visual detail implied by the Korean intent, plus mood/atmosphere but ONLY when the intent explicitly states it. Never invent an atmosphere the user did not name.

HARD RULES:
1. Output ONLY short phrases separated by commas. NEVER write full sentences. No subject+verb clauses, no period-terminated sentences, no markdown, no bullets, no numbering.
2. Each phrase is a few words at most (e.g. "wet cat ears", "soft rim light", "melancholic mood"). Noun phrases / adjective phrases only.
3. Do NOT restate anything the given tags already express. If a tag already covers it, leave it out. No paraphrases of existing tags.
4. Do NOT name characters or series. Character identity is already handled by the tags; never write "X from Y".
5. Concrete visual details may be included when implied by the Korean intent. But MOOD/ATMOSPHERE phrases (e.g. "melancholic mood", "tense atmosphere", "lonely feeling") are allowed ONLY when the Korean intent EXPLICITLY states that mood/emotion in words. Do NOT infer or invent an atmosphere from the scene — if the user did not name the mood, do not add one.
6. Do not invent content unrelated to the Korean intent. Stay grounded in what the intent implies.
7. English only. Output the bare comma-separated list and nothing else.
8. You are also given the meaning each tag already covers (COVERED MEANINGS). Treat these strictly as an exclusion list: any visual detail or nuance listed there is ALREADY expressed by the tags, so you must NOT write a phrase for it. Use them only to avoid duplication, never as material to describe."""


# ---------------------------------------------------------------------------
# 자연어(NL) 톤 프리셋
# ---------------------------------------------------------------------------
# 사용자가 NL 프롬프트의 장황함을 조절한다. SYSTEM_NL / SYSTEM_NL_MULTI 의
# {TONE_RULES} 슬롯에 주입되며, 톤별 권장 temperature 도 함께 정의한다.
#   - rich     : 기존 동작(2문장 이상, 묘사적). 하위호환 기본값.
#   - plain    : 평이/담백. 정보만, 수식어 금지.
NL_TONES = {
    "rich": {
        "temperature": 0.4,
        "rules": (
            "TONE: Write at least 2 sentences, descriptive and fluent. "
            "Avoid empty praise words like 'beautiful', 'cute', 'gorgeous', 'stunning', "
            "but you may use connective and descriptive phrasing freely."
        ),
    },
    "plain": {
        "temperature": 0.35,
        "rules": (
            "TONE: Plain and factual, but STILL WRITE REAL SENTENCES — this is the most "
            "important rule for this tone. Weave the keywords into one or more complete English "
            "sentences with subjects and verbs (e.g. 'A girl leans against a wall, glancing around "
            "warily as she checks her surroundings.'). "
            "NEVER output the keywords as a comma-separated list or a string of bare fragments — "
            "that defeats the purpose; the result must read as prose, not as tags. "
            "Keep it restrained: do NOT add decorative adjectives or mood words (no 'beautiful', "
            "'cute', 'gorgeous', 'stunning', 'elegant', 'delicate', 'ethereal', 'breathtaking', "
            "'mesmerizing', etc.), no metaphors, and no scene-setting beyond what the keywords imply. "
            "Simple, direct sentences are good — but they must be sentences. "
            "Still keep any '<Character> from <Series>' source structure intact."
        ),
    },
}
DEFAULT_NL_TONE = "rich"


def _tone(tone: str | None) -> dict:
    """톤 키 → 프리셋. 알 수 없으면 기본(rich)."""
    return NL_TONES.get((tone or DEFAULT_NL_TONE), NL_TONES[DEFAULT_NL_TONE])


# ---------------------------------------------------------------------------
# 공통 호출 헬퍼
# ---------------------------------------------------------------------------
def _build_payload(system: str, user: str, temperature: float) -> dict:
    """OpenAI 호환 chat completions payload 구성 (thinking 제어 포함)."""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "top_p": 0.9,
    }
    # thinking 비활성 (llama.cpp / vLLM 컨벤션)
    if LLM_THINKING_ENABLED:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    else:
        payload["thinking_budget_tokens"] = 0
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


async def _call(system: str, user: str, temperature: float, timeout: float = 120.0) -> str:
    """LLM 호출 후 content 문자열 반환."""
    payload = _build_payload(system, user, temperature)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(LLM_API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise LLMError(f"LLM 호출 실패: {e}")


async def _call_json(
    system: str, user: str, schema: dict, name: str,
    temperature: float = 0.0, seed: int = 1234, timeout: float = 180.0,
) -> dict:
    """
    JSON 스키마 강제 호출 (structured 파이프라인 전용, 기존 _call 과 별개).

    - response_format=json_schema: llama.cpp 가 스키마 문법으로 디코딩을 제한한다
      (코드펜스·뒷말 없이 스키마에 맞는 JSON 만 나옴. 서버 b12004 에서 확인).
    - thinking 은 항상 끈다: 선택 프롬프트(~2천 토큰)에서 사고가 6천 토큰을 넘겨
      슬롯 컨텍스트 8192 를 다 쓰고 본문이 비는 실패가 40% 나왔다(벤치 실측).
    - temperature 0 + 고정 seed. 프롬프트 캐시는 config.LLM_CACHE_PROMPT(기본 켬):
      켜면 시스템 프롬프트를 다시 처리하지 않아 호출당 3~5초 빠르다(이 서버는 프롬프트 처리 ~150~200 tok/s).
      대신 같은 입력 두 번 실행이 분해 8/10·선택 9/10 만 동일(점수는 같음). 끄면 10/10 동일(벤치 실측).
    - 캐시를 켠 호출에서 드물게(벤치 69회 중 1회, 영어 입력) 스키마를 벗어난 혼잣말이 출력 한도까지 반복됐다.
      같은 입력을 캐시 끄고 보내면 정상 → 응답이 비거나·JSON 이 깨지거나·한도에 걸리면 캐시를 끄고 1회 재시도.
    - 출력 한도 512: 분해·선택 출력은 100토큰 안팎이라 충분하고, 실패 시 낭비 시간을 줄인다(~38 tok/s).
    """
    try:
        return await _call_json_once(system, user, schema, name, temperature, seed, timeout, LLM_CACHE_PROMPT)
    except LLMError as e:
        if not LLM_CACHE_PROMPT:
            raise
        logger.warning(f"구조화 호출 실패 → 캐시 끄고 재시도 ({name}): {e}")
        return await _call_json_once(system, user, schema, name, temperature, seed, timeout, False)


async def _call_json_once(system, user, schema, name, temperature, seed, timeout, cache_prompt) -> dict:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": 512,
        "thinking_budget_tokens": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_prompt": cache_prompt,
        "seed": seed,
        "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema}},
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(LLM_API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise LLMError(f"LLM 호출 실패: {e}")
    choice = data["choices"][0]
    content = (choice["message"].get("content") or "").strip()
    finish = choice.get("finish_reason")
    if not content or finish == "length":
        raise LLMError(f"LLM 응답 본문 없음 또는 출력 한도 (finish={finish}, usage={data.get('usage')})")
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise LLMError(f"LLM JSON 파싱 실패: {e}; 앞부분={content[:200]!r}")


def _strip_markdown(text: str) -> str:
    """자연어 응답의 마크다운 코드블록/따옴표 제거."""
    text = re.sub(r'^```[\w]*\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
    return text.strip().strip('"')


# generate_nl_multi 입력에 쓰는 메타 라벨/플래그가 NL 출력에 그대로 베껴
# 나오는 누수를 후처리로 제거한다(3중 방어의 마지막 safety net). 새 누수 표현이
# 발견되면 이 목록에 패턴을 추가한다.
_META_LEAK_PATTERNS = [
    re.compile(r'\bCHARACTER\s*\d*\s*:', re.IGNORECASE),     # 블록 헤더
    re.compile(r'\bBACKGROUND\s*:', re.IGNORECASE),
    # key="value" / key=[...] / key=true 형태 라벨
    re.compile(r'\b(?:name|series|tags|description)\s*=\s*'
               r'(?:"[^"]*"|\[[^\]]*\]|true|false)', re.IGNORECASE),
    # 장황 라벨/괄호 안내문
    re.compile(r'original_character\s*=\s*true\s*(?:\([^)]*\))?', re.IGNORECASE),
    re.compile(r'\(\s*(?:new|unlisted)[^)]*character[^)]*\)', re.IGNORECASE),
    re.compile(r'\(\s*no\s+canon\s+name[^)]*\)', re.IGNORECASE),
    re.compile(r'\(\s*new/unlisted[^)]*\)', re.IGNORECASE),
    # 단축 플래그(입력에서 쓰는 형태)
    re.compile(r'\[\s*(?:original|new|unlisted|passthrough)\s*\]', re.IGNORECASE),
]


def _strip_meta_leak(text: str) -> str:
    """입력 메타 라벨이 NL 에 누수된 경우 제거하고 잔여 구두점/공백을 정리한다."""
    for pat in _META_LEAK_PATTERNS:
        text = pat.sub(' ', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\s+([,.;:])', r'\1', text)
    text = re.sub(r'([,;:])\s*\1+', r'\1', text)
    text = re.sub(r'^\s*[,;:.]+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip().strip(',;:').strip()


# ---------------------------------------------------------------------------
# 단계별 함수
# ---------------------------------------------------------------------------
async def generate_nl_prompt(
    korean_prompt: str,
    keywords: list[str],
    temperature: float | None = None,
    tone: str | None = None,
) -> str:
    """
    [자연어 생성] Anima용 자연어 영어 프롬프트.

    tone: rich(기존)/plain(담백). {TONE_RULES} 슬롯에 룰 주입.
    temperature: None 이면 톤별 권장 온도 사용. 값이 오면(UI 슬라이더 등) 그 값 우선.
    """
    preset = _tone(tone)
    temp = preset["temperature"] if temperature is None else temperature
    system = SYSTEM_NL.replace("{TONE_RULES}", preset["rules"])
    user = f"Korean intent: {korean_prompt}\nKeywords: {', '.join(keywords)}"
    try:
        content = await _call(system, user, temp)
        return _strip_markdown(content)
    except LLMError:
        return ""  # 자연어는 보조 수단이라 실패해도 전체를 막지 않음


async def select_character_and_series(
    user_name: str,
    user_series: str,
    char_candidates: list[dict],
    series_candidates: list[dict],
    temperature: float = 0.0,
) -> tuple[str, str]:
    """
    [분할입력 캐릭터/작품 선택] 사용자 입력 + DB 후보군을 LLM에 주고 올바른 태그 선택.

    char_candidates / series_candidates 각 원소: {"tag": str, "score": float, "aliases": list[str]}
    Returns: (character_tag, series_tag)  — 못 고르면 빈 문자열.
    """
    if not user_name and not user_series:
        return "", ""

    def _fmt(cands: list[dict]) -> str:
        lines = []
        for c in cands:
            al = ", ".join(c.get("aliases", [])[:6])
            lines.append(
                f'- {c["tag"]} (score={c.get("score", 0):.3f}'
                + (f', aliases=[{al}]' if al else "")
                + ")"
            )
        return "\n".join(lines) if lines else "(none)"

    user = (
        f'CHARACTER name: "{user_name}"\n'
        f'SERIES name: "{user_series}"\n\n'
        f"CHARACTER_CANDIDATES:\n{_fmt(char_candidates)}\n\n"
        f"SERIES_CANDIDATES:\n{_fmt(series_candidates)}"
    )

    try:
        content = await _call(SYSTEM_SELECT_CHARACTER, user, temperature)
    except LLMError:
        return "", ""

    # JSON 객체 파싱 (마크다운 펜스 제거 후)
    txt = _strip_markdown(content)
    import json, re as _re
    m = _re.search(r"\{.*\}", txt, _re.DOTALL)
    if not m:
        return "", ""
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return "", ""

    char_tag = (obj.get("character") or "").strip()
    series_tag = (obj.get("series") or "").strip()

    # 환각 방지: 후보에 실제 존재하는 태그만 채택
    char_set = {c["tag"] for c in char_candidates}
    series_set = {c["tag"] for c in series_candidates}
    if char_tag not in char_set:
        char_tag = ""
    if series_tag not in series_set:
        series_tag = ""
    return char_tag, series_tag


async def generate_nl_multi(
    characters: list[dict],
    background_tags: list[str] | None = None,
    background_desc: str = "",
    temperature: float | None = None,
    tone: str | None = None,
) -> str:
    """
    [분할입력 자연어] 인물 블록 + 배경으로 인물 단위 자연어 프롬프트 생성.

    Args:
        characters: 인물 블록 리스트. 각 원소:
            {
              "name": str,        # 사용자가 적은 캐릭터명(공백 가능)
              "series": str,      # 작품명(공백 가능)
              "tags": list[str],  # 그 인물 칸에서 확정된 태그(캐릭터/작품/속성)
              "desc": str,        # 원본 캐릭터묘사(한국어, 폴백 설명용)
            }
        background_tags: 배경칸에서 확정된 태그
        background_desc: 배경칸 원본(한국어)

    이름만 나열되지 않도록 SYSTEM_NL_MULTI 가 각 인물에 외형묘사를 강제한다.
    실패 시 빈 문자열(자연어는 보조 수단).
    """
    lines: list[str] = []
    for i, ch in enumerate(characters, 1):
        parts = [f"CHARACTER {i}:"]
        if ch.get("is_original"):
            # 오리지널 캐릭터: 작품/캐릭터명 없음. LLM이 임의 이름을 지어 붙인다.
            # 단축 플래그로 전달(장황 안내문이 NL 에 베껴지는 누수 방지). 의미는
            # SYSTEM_NL_MULTI RULE 3/10 이 정의한다.
            parts.append("[original]")
        else:
            if ch.get("name"):
                parts.append(f'name="{ch["name"]}"')
            if ch.get("series"):
                parts.append(f'series="{ch["series"]}"')
            if ch.get("is_passthrough"):
                # 패스스루: DB에 아직 없는 신규 캐릭터를 사용자가 직접 지정.
                # 매칭된 캐릭터 태그가 없어도 이 이름을 그대로 캐릭터 핸들로 써야 한다.
                # 단축 플래그로 전달(안내문 누수 방지). 의미는 RULE 2/10 이 정의한다.
                parts.append("[new]")
        if ch.get("tags"):
            parts.append(f'tags=[{", ".join(ch["tags"])}]')
        if ch.get("desc"):
            parts.append(f'description="{ch["desc"]}"')
        lines.append("  ".join(parts))

    if background_tags or background_desc:
        bparts = ["BACKGROUND:"]
        if background_tags:
            bparts.append(f'tags=[{", ".join(background_tags)}]')
        if background_desc:
            bparts.append(f'description="{background_desc}"')
        lines.append("  ".join(bparts))

    preset = _tone(tone)
    temp = preset["temperature"] if temperature is None else temperature
    system = SYSTEM_NL_MULTI.replace("{TONE_RULES}", preset["rules"])
    user = "\n".join(lines)
    try:
        content = await _call(system, user, temp)
        return _strip_meta_leak(_strip_markdown(content))
    except LLMError:
        return ""


# ---------------------------------------------------------------------------
# 단어형(phrase) 보조 출력
# ---------------------------------------------------------------------------
PHRASE_TEMPERATURE = 0.25   # 잔차만 뽑으므로 낮게(결정적). plain(0.3)보다 더 보수적.


def _phrase_key(s: str) -> str:
    """phrase/태그 중복 비교용 정규화: 소문자 + 언더스코어/공백 통일."""
    return re.sub(r'[\s_]+', ' ', s.lower()).strip()


def _postprocess_phrases(content: str, final_tags: list[str]) -> str:
    """
    LLM 출력을 'a, b, c' 콤마 나열로 정규화하고 final_tags 와의 중복을 코드로 제거한다.

    프롬프트가 중복 회피를 지시하지만 모델이 어길 수 있으므로(태그명을 그대로 뱉는 등),
    여기서 문자열 차원의 중복(대소문자/언더스코어 차이 포함)을 마지막으로 보증한다.
    """
    content = re.sub(r'^```[\w]*\n?', '', content)
    content = re.sub(r'\n?```$', '', content).strip()
    tagkeys = {_phrase_key(t) for t in final_tags}
    out: list[str] = []
    seen: set[str] = set()
    for piece in re.split(r'[,\n]', content):   # 콤마 또는 줄바꿈 구분
        tok = piece.strip().strip('.;')
        tok = re.sub(r'^[-*0-9.)\s]+', '', tok)  # 앞 불릿/번호 제거
        tok = re.sub(r'\s+', ' ', tok).strip()
        if not tok:
            continue
        k = _phrase_key(tok)
        if k in tagkeys or k in seen:           # 태그 중복 / 내부 중복 제거
            continue
        seen.add(k)
        out.append(tok)
    return ", ".join(out)


def _build_covered_meanings(final_tags: list[str], defs: dict[str, dict]) -> str:
    """단어형+ 용: 최종 태그가 이미 커버하는 의미(definition/aliases/분류)를 텍스트로."""
    lines: list[str] = []
    for t in final_tags:
        meta = defs.get(t)
        if not meta:
            continue
        bits: list[str] = []
        if meta.get("major") or meta.get("minor"):
            bits.append("/".join(x for x in (meta["major"], meta["minor"]) if x))
        if meta.get("definition"):
            bits.append(meta["definition"])
        if meta.get("aliases"):
            bits.append(", ".join(meta["aliases"]))
        if bits:
            lines.append(f"- {t}: " + " | ".join(bits))
    return "\n".join(lines)


async def generate_phrase(
    korean_prompt: str,
    final_tags: list[str],
    defs: dict[str, dict] | None = None,
    temperature: float | None = None,
) -> str:
    """
    [단어형 보조출력] 확정 태그가 표현 못 한 잔차를 짧은 영어 구로 나열.

    완전한 문장(NL)이 길어질수록 그림체가 흔들리는 문제의 대안. NL 칸을 대체한다.
    확정 태그의 의미 범위(COVERED MEANINGS)를 배제 목록으로 줘서, 태그가 이미
    표현한 부분과 겹치지 않는 잔차(시각 디테일/분위기)만 뽑게 한다.

    Args:
        korean_prompt: 원본 한국어 의도.
        final_tags:    3단계까지 확정된 최종 Danbooru 태그(중복 배제 기준).
        defs:          get_definitions(variant) 결과(태그→의미). 배제 목록 구성에 사용.

    실패 시 빈 문자열(보조 수단이라 전체를 막지 않음).
    """
    temp = PHRASE_TEMPERATURE if temperature is None else temperature

    user_lines = [
        f"Korean intent: {korean_prompt}",
        f"Existing tags (do NOT restate these): {', '.join(final_tags)}",
    ]
    if defs:
        covered = _build_covered_meanings(final_tags, defs)
        if covered:
            user_lines.append("COVERED MEANINGS (already expressed by the tags):")
            user_lines.append(covered)
    user = "\n".join(user_lines)

    try:
        content = await _call(SYSTEM_PHRASE, user, temp)
        return _postprocess_phrases(_strip_markdown(content), final_tags)
    except LLMError:
        return ""
