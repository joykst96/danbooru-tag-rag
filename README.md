# danbooru-tags-rag

한국어 묘사를 받아 Danbooru 태그와 자연어 프롬프트를 만드는 로컬 도구입니다.
로컬 임베딩(`google/embeddinggemma-2`)과 로컬 LLM(llama.cpp)으로 동작하고, DB에 없는 태그는 내보내지 않습니다.

A local Korean→Danbooru tag generator built on a local embedding model (`google/embeddinggemma-2`)
and a local LLM (llama.cpp). Tags that are not in the database never reach the output.

---

## 한국어

### 풀려는 문제

언어모델로 한국어를 영어 태그로 바꾸면 두 가지가 깨집니다.

1. **환각**: DB에 없는 태그를 지어냅니다. (예: 은발 → `silver_hair`. Danbooru엔 그 태그가 없고 `grey_hair`로 통합돼 있습니다.)
2. **세부 표현 붕괴**: "머리 양쪽 드래곤 뿔" 같은 디테일을 큰 단어 하나로 뭉갭니다.

이 도구는 둘 다 구조로 막습니다.

- **환각 차단**: 선택 단계는 후보 **번호**만 고를 수 있는 JSON 스키마로 강제합니다. 그래서 후보 밖 태그는 구조적으로 나올 수 없습니다. 그 뒤에 DB 실존 확인과 `_(cosplay)` 같은 파생 태그 제거를 한 번 더 거칩니다.
- **세부 보존**: 문장을 개념 하나짜리 조각으로 나눠 조각마다 따로 검색하고, 실제 후보 중에서 원문 의도에 맞는 것을 LLM이 고릅니다. 맞는 후보가 없으면 조각을 버리지 않고 짧은 영어 구로 남깁니다.

### 동작 흐름 (기본 모드)

```
입력(한국어)
 1. 구조화 분해 (JSON 스키마 강제)  → 인원 / 캐릭터 / 조각[{한국어, 영어}]  (조각 하나 = 개념 하나)
 2. 조각별 후보                    → 어휘 일치 + 한국어 질의 top5 + 영어 질의 top5  (카테고리 where 사전 필터)
 3. 조각별 선택 (후보 번호만 허용)  → 맞는 후보가 없을 때만 짧은 영어 구
 4. 최종 = 인원수 태그 + 캐릭터 태그 + 선택 태그·구  (+ DB 실존·cosplay 이중 방어)
 5. 자연어 프롬프트 생성 (톤: 묘사적/담백/단어형+)
```

- **분해**: 인원(girls/boys/others)은 코드가 `1girl`/`2girls`/`1boy`… 로 바꿉니다(동물은 사람이 아님, 이름·관계어로 성별 추론). 캐릭터·작품 이름은 따로 받아 캐릭터·작품 카테고리에서만 찾고, 나머지는 일반 카테고리에서 찾습니다. 그래서 "은발"이 특정 캐릭터 태그로 새는 오염이 없습니다.
- **문자열 제약**: 분해·선택 출력의 문자열에는 대괄호와 큰따옴표를 쓸 수 없게 스키마 패턴으로 막습니다. QAT Q4_K_M 모델에서 쌍 구분자가 문자열 안으로 새어 조각이 사라지는 붕괴가 있었는데(새 장면 10개 중 2~3개), 이 패턴으로 없어졌습니다. 정상 문장의 출력은 패턴 전후가 글자 단위로 같습니다.
- **임베딩**: `google/embeddinggemma-2`(텍스트 인코더만, 768차원) + 인덱스 `sim`입니다.
  - `sim`은 새 한국어 소스(`mengod-hf/danbooru-tags-kr`)의 정의·별칭으로 `태그명 정의 별칭` 문서를 만들고, 문서와 쿼리 모두 `SentenceSimilarity` 프롬프트(`task: sentence similarity | query: `)를 붙여 임베딩합니다.
  - 평가셋 168문항에서 단일 질의 MRR 0.739, 서비스 방식(조각별 한국어+영어+어휘 일치) 후보 회수율 96%입니다(근거: tag-embedding 벤치).
- **정확도**(같은 LLM gemma4 26B A4B, 실행 전에 고정한 기준):
  - 새 장면 10개 필수 개념 46/48(96%).
  - 인원수: 기존 40문장 한국어·영어 39/40, 실행 전에 고정한 새 문장 24개 한국어 21/24·영어 23/24.
- **속도**: 요청 한 번에 LLM을 3번 부릅니다(분해, 선택, 자연어). 운영 서버(MoE 일부 CPU)에서 요청 전체가 약 14~19초입니다.
  - 프롬프트 캐시(`LLM_CACHE_PROMPT`, 기본 on)를 켜서 시스템 프롬프트 재처리를 없앴습니다(호출당 3~5초).
  - 출력 JSON을 압축했고(선택 약 34토큰, 분해 약 80토큰), 질의 임베딩을 한 번에 묶습니다.
  - 캐시를 켜면 같은 입력을 두 번 돌렸을 때 결과가 조금 다를 수 있습니다(채점 점수는 같음). `LLM_CACHE_PROMPT=off`면 같은 결과가 나오지만 호출당 3~5초 느립니다.
  - 응답이 비거나 출력 한도에 걸리면 캐시를 끄고 한 번 재시도합니다.
- **단계별 스트리밍**: 각 단계가 끝나는 즉시 UI에 표시합니다. 동시 요청은 1개씩 직렬 처리하며 대기 순번을 실시간으로 보여 줍니다.

### 두 가지 입력 모드

- **기본 모드**: 한 문장을 통째로 넣습니다. 위 흐름을 그대로 탑니다. **태그 검색 범위**(일반/작가/작품/캐릭터/메타)로 일반 조각의 검색 카테고리를 고를 수 있습니다. 기본은 일반(cat 0)이고, 일반 외 카테고리를 켜면 품질이 떨어질 수 있습니다.
- **고급(분할) 모드**: 인물칸 여러 개 + 배경칸 하나로 나눠 입력합니다. 다인물 장면에서 인물이 섞이지 않게 하고, 인물 단위 자연어를 만듭니다. 인물칸마다 **캐릭터 출처**를 고릅니다.
  - **DB에서 찾기**: 작품·캐릭터 후보를 DB에서 받아 LLM이 입력 기준으로 고릅니다(동명이인·같은 작품 조연 구분). 작품 태그는 자연어에만 쓰고 최종 태그엔 넣지 않습니다.
  - **DB에 없는 캐릭터(패스스루)**: 검색을 건너뛰고 입력한 이름을 자연어와 태그 맨 앞에 그대로 씁니다.
  - **오리지널**: 작품·캐릭터 검색을 생략하고 자연어에서 임의 이름을 붙입니다.
  - 묘사칸·배경칸은 기본 모드와 같은 분해·검색·선택을 거칩니다(일반 조각만). 칸에 적힌 인물 이름은 조각에서 지워 `xxx_(cosplay)` 오염을 막습니다.
  - 인원수는 인물칸 하나를 한 명으로 보고, 칸마다 분해로 성별을 정해 합산합니다(소녀 두 칸 → `2girls`).

### 자연어 톤

자연어의 길이·문체를 프리셋으로 고릅니다. 톤을 고르면 온도가 권장값으로 맞춰지고, 고급 설정에서 직접 바꿀 수 있습니다(분해·선택 단계는 항상 온도 0).

- **묘사적**(0.4): 2문장 이상, 유려한 묘사.
- **담백**(0.35): 수식어 없이 짧고 직접적인 문장. 확정 태그가 아니라 태그로 수렴하기 전의 영어 조각을 입력으로 받아, 수렴하며 잘린 의미를 되살립니다(캐릭터 정체성은 유지).
- **단어형+**(0.25): 문장 대신 짧은 영어 구를 쉼표로 나열합니다. 문장이 길수록 그림체가 흔들리는 문제의 대안입니다. 확정 태그가 표현하지 못한 잔차(시각 디테일, 명시된 분위기)만 담고, 태그의 의미 범위를 제외 목록으로 줘 중복을 줄입니다. 결과는 뱃지로 표시되고 휠로 가중치를 줄 수 있습니다. 고급 모드에서도 같은 방식입니다.

캐릭터 태그가 있으면 자연어가 `<캐릭터> from <작품>, ...` 형식으로 시작합니다. 이 구조는 톤과 관계없이 유지됩니다.

### 가중치 표기 / 고정 태그 / 태그 편집

- **가중치 표기(로컬 / NAI)**: 가중치를 로컬(`(tag:weight)`) 또는 NAI(`weight::tag::`) 형식으로 출력합니다. NAI는 음수 가중치를 지원합니다(하한 -10). 내부 상태는 로컬 표기로 저장하고 표시·복사할 때만 바꾸므로, 휠로 다시 조절해도 표기가 겹치지 않습니다.
- **고정 태그**: 퀄리티·작가 태그를 LLM·DB를 거치지 않고 결과에 붙입니다. 각각 **맨앞/맨뒤**를 고르고, 같은 쪽이면 **우선순위**로 순서를 정합니다. 출력 순서는 `[앞 고정] + [패스스루 + Danbooru 태그] + [뒤 고정]`입니다. 작가 태그는 `없음/@/artist:` 접두사를 지원하고, 고정 태그는 언더스코어 치환·괄호 이스케이프를 받지 않습니다.
- **결과 태그 직접 편집**: textarea에서 태그를 고칠 수 있습니다(뱃지와 양방향 동기화). 뱃지는 마우스 휠로 가중치(0.1 단위)를 조절합니다.

### 로그

- **콘솔 로그**: 수신(📥)·대기 인원, 모드(🟢 일반 / 🔵 고급)·입력(📝)·설정(⚙️ 톤/검색범위/자연어 온도)·최종 태그(✨), 실패(❌)를 실시간으로 찍습니다(`docker compose logs -f`).
- **파일 로그**: 입력 / 최종 태그(언더스코어→공백) / 자연어를 JSON Lines로 일자별 저장합니다(`logs/`). 정상 완료 시에만 기록합니다.

### 준비

1. 의존성 설치: `uv sync` 또는 `pip install -e .`
2. `.env` 작성 (LLM 엔드포인트, `EMBEDDING_DEVICE`) — `.env.example` 참고
3. 태그 CSV 생성: `python -m core.tags_source` → 루트 `danbooru-tags.csv` (HF `mengod-hf/danbooru-tags-kr` 2026-11-10판, post_count ≥ 50, 118,133개)
4. 임베딩 모델 다운로드(최초 1회): `python download_model.py` → `models/embeddinggemma-2`
5. 인덱스 빌드: `python -m core.builder` → `data/embeddinggemma-2/lancedb_sim` (GPU 권장, CPU는 매우 느림).

> Python 3.11 이상, sentence-transformers 6.1 이상, transformers 5.22 이상이 필요합니다(embeddinggemma-2).
>
> torch는 GPU 환경에 맞는 CUDA 빌드를 따로 설치합니다(예: cu130). `lancedb` 외에 `pylance`와 `pandas`가 필요합니다(pyproject에 있음).
>
> LLM 서버는 `response_format: json_schema`(문법 제한 디코딩, 문자열 `pattern` 포함)를 지원해야 합니다. llama.cpp b12004·b12192에서 확인했습니다.

### 사용

- API 서버: `python -m core.api` (포트 3333)
- 브라우저: `http://localhost:3333`
- 단일 쿼리 직접 검색: UI 우측 패널 또는 `/api/direct_search`

### 배포

Docker 기반입니다. 자세한 절차는 [DEPLOY.md](DEPLOY.md)를 보세요.

```bash
docker compose up -d --build
curl http://localhost:3333/api/health
```

모델·LanceDB·CSV·`.env`는 이미지에 굽지 않고 볼륨으로 마운트합니다. 인덱스는 GPU 머신에서 빌드해 복사합니다(벡터는 디바이스와 무관). `index.html`과 `web/`은 이미지에 COPY되므로 UI를 고치면 `--build`로 다시 빌드해야 합니다.

### 구조

| 파일 | 역할 |
|------|------|
| `core/config.py` | 전역 설정 (`.env` 로드) |
| `core/parser.py` | description 타입별 분기 파싱 + 임베딩 텍스트(`태그명 정의 별칭`) |
| `core/tags_source.py` | 태그 소스(HF mengod-hf/danbooru-tags-kr, 고정 revision) → `danbooru-tags.csv` 생성 |
| `core/embeddings.py` | embeddinggemma-2 임베딩 (토큰 예산 배치) |
| `core/database.py` | LanceDB (카테고리 필터 검색) |
| `core/builder.py` | CSV → `sim` 인덱스 빌드 |
| `core/search.py` | 벡터 검색, 사전 필터·어휘 일치, 실존 태그 집합, 이름 오염 차단 |
| `core/llm.py` | LLM 호출 (구조화 호출 `_call_json`, 자연어·톤, 단어형, 캐릭터 선택) |
| `core/llm_structured.py` | 구조화 분해 / 조각별 선택 프롬프트·스키마 |
| `core/pipeline_structured.py` | 기본 모드 파이프라인 + 고급 모드 칸 처리(`describe`) |
| `core/pipeline_split.py` | 고급(분할) 모드 파이프라인 (인물칸/배경칸) |
| `core/genlog.py` | 생성 로그 (콘솔 + 파일 JSONL) |
| `core/api.py` | FastAPI + 정적 서빙(`/`, `/web/…`) + 대기열 |
| `index.html` | UI 마크업 |
| `web/css/` | 디자인 토큰(`tokens.css`, 생성 파일) + 앱 스타일(`base`, `layout`, `controls`, `results`, `app-tokens`) |
| `web/js/` | UI 모듈 (`main`, `api`, `generate`, `results`, `input-form`, `fixed-tags`, `direct-search`, `autocomplete`, `tooltip` 등) |

---

## English

### What it solves

Translating Korean to English tags with an LLM breaks two ways:

1. **Hallucination**: it invents tags that are not in the DB (e.g. "은발" → `silver_hair`, which doesn't exist; Danbooru merges it into `grey_hair`).
2. **Detail collapse**: it flattens fine detail ("dragon horns on both sides of the head") into a head term.

Both are blocked by construction.

- **No hallucinated tags**: the selection step may only return candidate **numbers** (JSON-schema constrained), so off-list tags cannot be produced. A DB-existence check and removal of derived tags such as `_(cosplay)` follow.
- **Detail preservation**: the sentence is split into one-concept units, each retrieved separately, and the LLM picks from real candidates against the original intent. When nothing matches, the unit is kept as a short English phrase instead of being dropped.

### Flow (basic mode)

1. Structured decomposition (JSON-schema constrained) → people / characters / units `{ko, en}` (one concept per unit)
2. Per-unit candidates → exact tag-name match + Korean query top5 + English query top5 (category `where` prefilter)
3. Per-unit selection by candidate number only → a short English phrase only when no candidate matches
4. Final = headcount tags + character tags + selected tags/phrases (+ DB-existence and cosplay double guard)
5. Natural-language prompt (descriptive / plain / phrase+)

Strings in the decomposition and selection output may not contain brackets or double quotes (schema `pattern`). With the QAT Q4_K_M model, pair delimiters sometimes leaked into the strings and units vanished (2-3 of 10 new scenes); the pattern removes this, and outputs for normal sentences are byte-identical with and without it.

Embedding is `google/embeddinggemma-2` (text encoder only, 768-d) with index `sim`: `<tag> <definition> <aliases>` built from the `mengod-hf/danbooru-tags-kr` source, with the `SentenceSimilarity` prompt on both documents and queries (single-query MRR 0.739; candidate recall 96% in the service setting). With gemma4 26B A4B, required-concept coverage on 10 new scenes is 46/48, headcount accuracy is 39/40 on a 40-sentence set and 21/24 (Korean) / 23/24 (English) on a fresh held-out set, and a full request takes about 14-19 s on the production server thanks to prompt caching (`LLM_CACHE_PROMPT`, default on), compact JSON output and batched query embedding.

### Two input modes

**Basic mode**: one sentence through the flow above. A **tag search scope** selector (general/artist/copyright/character/meta) sets the categories searched for general units; the default is general, and other categories can degrade quality.

**Advanced split mode**: per-character fields plus one background field. Each character picks a **source**: *DB lookup* (the LLM picks character/series tags from DB candidates; series tags feed NL only), *unlisted character (passthrough)* (skips search and uses the typed name verbatim), or *original* (skips search; NL invents a name). Description and background fields go through the same decomposition, retrieval and selection (general units only), with character names stripped to avoid `_(cosplay)` pollution. Each character field counts as one person; its gender comes from the decomposition and the counts are summed (two girl fields → `2girls`).

**Natural-language tone** presets set verbosity and a recommended temperature (descriptive 0.4, plain 0.35, phrase+ 0.25), adjustable under advanced settings; decomposition and selection always run at temperature 0. *Plain* is fed the pre-convergence English units to recover meaning lost to tag convergence; *phrase+* emits short comma-separated phrases carrying only what the tags don't express, in both modes. When a character tag is present, NL opens with `<Character> from <Series>, ...` regardless of tone.

**Weight syntax (local / NAI)**, **fixed tags** (quality and artists, front/back position and priority) and **editable result tags** work as described in the Korean section.

### Setup / Run

```bash
uv sync                      # or pip install -e .
# .env (see .env.example)
python download_model.py     # once → models/embeddinggemma-2
python -m core.tags_source   # → danbooru-tags.csv (mengod-hf/danbooru-tags-kr 2026-11-10 edition, post_count >= 50)
python -m core.builder       # GPU recommended → data/embeddinggemma-2/lancedb_sim
python -m core.api           # port 3333
```

torch must be installed as a CUDA build matching your GPU. `pylance` and `pandas` are required alongside `lancedb`. The LLM server must support `response_format: json_schema` including string `pattern` (checked on llama.cpp b12004 and b12192).

### Deploy

Docker-based; see [DEPLOY.md](DEPLOY.md). Models, LanceDB, CSV and `.env` are volume-mounted, not baked into the image. Build the index on a GPU machine and copy it over. `index.html` and `web/` are COPYed into the image, so rebuild with `--build` after UI changes.

### License

MIT
