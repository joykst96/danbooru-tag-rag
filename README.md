# danbooru-tags-rag

한국어 입력을 받아 cross-lingual 검색으로 Danbooru 태그와 자연어 프롬프트를 생성하는 로컬 도구.
로컬 임베딩(multilingual-e5-large) + 로컬 LLM(llama.cpp) 기반입니다. 환각 없이 동작합니다.

A local Korean→Danbooru tag generator using cross-lingual retrieval over local embeddings
(multilingual-e5-large) and a local LLM (llama.cpp). Hallucination-free by construction.

---

## 한국어

### 풀려는 문제

언어모델로 한국어를 영어 태그로 바꾸면 두 가지가 깨집니다.

1. **환각** — DB에 없는 태그를 지어냅니다. (예: 은발 → `silver_hair`. Danbooru엔 그 태그가 없고 `grey_hair`로 통합돼 있습니다.)
2. **세부 표현 붕괴(mode collapse)** — "머리 양쪽 드래곤 뿔" 같은 디테일을 head 단어로 뭉갭니다.

이 도구는 둘 다 **구조적으로** 막습니다.

- **환각 차단**: 최종 태그는 (1) LLM 프롬프트로 "후보 외 금지" 강제 + (2) 후보풀 대조 + (3) DB 실존 태그 코드필터, 삼중으로 거릅니다. LLM이 무엇을 생성하든 DB에 없으면 코드가 버립니다.
- **세부 보존**: 단순 번역이 아니라 한국어 의미단위로 분해해 각각 검색하고, 실존 후보 중에서 원본 의도에 맞는 것을 LLM이 선택합니다.

### 동작 흐름 (기본 모드, v3 — 2026-11-05~)

```
입력(한국어)
 1. 구조화 분해 (JSON 스키마 강제)  → 인원 / 캐릭터 / 조각[{한국어, 영어}]  (조각 하나 = 개념 하나)
 2. 조각별 후보                    → 어휘 일치 + 한국어 질의 top5 + 영어 질의 top5  (카테고리 where 사전 필터)
 3. 조각별 선택 (후보 번호만 허용)  → 맞는 후보가 없을 때만 짧은 영어 구
 4. 최종 = 인원수 태그 + 캐릭터 태그 + 선택 태그·구  (+ DB 실존·cosplay 이중 방어)
 5. 자연어 프롬프트 생성 (톤: 묘사적/담백/단어형+)
```

- **임베딩**: `google/embeddinggemma-2`(텍스트 인코더만, 768차원) + 인덱스 `nt`입니다. `nt`는 새 한국어 소스(`mengod-hf/danbooru-tags-kr`)의 정의·별칭을 모델 카드 형식(`title: 태그명 | text: 정의 별칭`)으로 임베딩합니다. CPU 쿼리는 e5-large보다 빠릅니다(56ms vs 103ms).
- **왜 바꿨나**: tag-embedding 벤치에서 같은 LLM(gemma4 26B A4B)과 같은 10문장으로 비교했습니다.
  - 필수 개념 충족률이 기존 4회 분해 82%에서 v3 95%로 올랐습니다(같은 조건 Claude 95~98%).
  - 같은 입력을 두 번 돌리면 결과가 10/10 같습니다(temperature 0, `cache_prompt=false`, 고정 seed).
  - 기존 방식의 실패 사례는 세 가지였습니다.
    - "단발머리 소년"처럼 인물에 붙은 속성이 인원수 조각으로 통째로 빠짐
    - 후보 밖 태그를 지어냄(`silver_hair`)
    - 근접 오답(`arm_armor`, `hat_loss`)
- **되돌리기**: `.env`의 `EMBEDDING_PROFILE=e5`로 e5-large와 기존 4회 분해 파이프라인(아래)으로 그대로 돌아갑니다. 파이프라인만 바꾸려면 `PIPELINE=decomposed`를 씁니다.

기존 4회 분해 파이프라인(`PIPELINE=decomposed`, e5 프로필의 기본값):

```
입력(한국어)
 0. 캐릭터 해석          → (작품?, 캐릭터) 쌍 추출 → cat3/4 검색 → 일괄 판별
                          (캐릭터가 없으면 건너뜀)
 1. 한국어 분해 + 검색    → 한국어 단위별 후보
 2. 통번역 분해 + 검색    → 영어 단위별 후보  (문맥 보존 위해 통째로 번역 후 분해)
 (한국어/영어 후보풀 합본)
 3. 태그 완성            → 원본 기준 선별 + 환각 코드필터 + 인원수 보호
 4. 자연어 프롬프트 생성
```

- **캐릭터 해석(0단계)**: 한 문장에 캐릭터 이름이 있으면(예: "원신 라이덴 쇼군이 ...") 별도 입력 없이도 자동으로 잡습니다. (작품?, 캐릭터) 쌍을 LLM이 추출해 캐릭터·작품 카테고리에서 검색하고, 후보군을 한 번에 LLM이 판별합니다. 확정된 캐릭터 태그는 최종 출력 맨 앞에 오고, 작품 태그는 자연어에만 쓰며, 추출된 이름은 일반 검색에서 제거해 `xxx_(cosplay)` 류 오염을 막습니다. 캐릭터가 없으면 이 단계를 건너뛰어 기존 흐름 그대로 돕니다.
- **2분할 DB**: 일반(cat 0) / 캐릭터·작품(cat 3,4)을 분리 검색합니다. 데이터셋이 작아 캐릭터 태그가 속성 검색을 오염시키는 현상(은발 → 특정 캐릭터)을 카테고리 필터로 차단합니다. 모든 카테고리(작가 1·메타 5 포함)를 인덱싱하되 프롬프트 파이프라인은 필요한 카테고리만 조회합니다(임베딩 벡터는 태그별 독립이라 상호 오염이 없습니다). v3는 LanceDB `where` 사전 필터를 씁니다. 기존 경로(`TagDatabase.search`)는 top_k×4를 가져와 거르는 사후 필터라서, 상위권을 다른 카테고리가 채우면 결과가 빌 수 있습니다(예: "여기사" → 로마자 작가명이 상위 20개를 채워 cat 0 결과 0건).
- **인원수 보호 / 정렬**: 검색으로 확정하되 LLM 선별 단계에서 인원수 슬롯을 따로 보호해 다인물 장면이 solo로 뭉개지는 것을 막고, 인원수 태그(1girl, 2girls 등)를 최종 출력 맨 앞에 배치합니다.
- **단계별 스트리밍**: 각 단계가 끝나는 즉시 UI에 표시합니다(키워드 먼저 보이고 최종은 나중). 동시 요청은 1개씩 직렬 처리하며 대기 순번을 실시간 표시합니다.

### 두 가지 입력 모드

- **기본 모드**: 한 문장을 통째로 넣습니다. 위 파이프라인을 그대로 탑니다. **태그 검색 범위**(일반/작가/작품/캐릭터/메타)를 골라 검색 대상 카테고리를 조절할 수 있습니다. 기본은 일반(cat 0)이며, 일반 외 카테고리를 켜면 출력 품질이 떨어질 수 있습니다.
- **고급(분할) 모드**: 인물별 칸 + 배경칸으로 나눠 입력합니다. 다인물 장면에서 인물이 섞이는 것을 막고 이름 기반 자연어 프롬프트를 만듭니다. 인물칸마다 **캐릭터 출처**를 3택으로 고릅니다.
  - **DB에서 찾기**: 작품·캐릭터 태그를 DB 후보군에서 LLM이 사용자 입력 기준으로 선택합니다(동명이인·동일작품 조연 구분). 작품 태그는 캐릭터 추론·자연어에만 쓰고 최종 출력엔 넣지 않습니다.
  - **DB에 없는 캐릭터(패스스루)**: DB에 아직 없는 신규 캐릭터. 검색을 건너뛰고 입력한 이름을 자연어와 태그 맨 앞에 그대로 씁니다.
  - **오리지널**: 작품/캐릭터 검색을 생략하고 자연어에서 임의 이름을 부여합니다.
  - 묘사칸·배경칸에 적힌 인물 이름은 검색에서 제거해 `xxx_(cosplay)` 류 오염을 막습니다.

### 자연어 톤

자연어 프롬프트의 장황함을 프리셋으로 조절합니다(온도는 톤에 맞춰 자동 설정, 이후 직접 미세조정 가능).

- **묘사적**: 2문장 이상, 유려한 묘사.
- **담백**: 수식어 없이 정보만, 짧고 직접적인 문장. 입력은 확정 태그가 아니라 DB 수렴 전 영어 분해단위를 받아, 태그로 수렴하며 잘려나간 의미를 복원합니다(캐릭터 정체성은 보존).
- **단어형+**: 문장 대신 짧은 영어 구를 쉼표로 나열합니다. 문장이 길수록 그림체가 흔들리는 문제의 대안으로, 확정 태그가 표현하지 못한 잔차(시각 디테일·명시된 분위기)만 담고 태그와 중복되지 않습니다. 태그의 의미 범위(definition)를 배제맥락으로 LLM에 줘 중복을 줄입니다. 출력은 결과 영역에서 뱃지로 표시되며 휠로 가중치를 줄 수 있습니다.

캐릭터 태그가 있으면 자연어가 `<캐릭터> from <작품>, ...` 형식으로 시작합니다(작품이 있을 때). 출처 구조는 톤과 무관하게 보존됩니다.

### 가중치 표기 / 고정 태그 / 태그 편집

- **가중치 표기(로컬 / NAI)**: 모든 가중치를 로컬(`(tag:weight)`) 또는 NAI(`weight::tag::`) 형식으로 출력합니다. NAI는 음수 가중치를 지원합니다(하한 -10). 내부 상태는 항상 로컬 표기로 저장하고 표시·복사 시에만 변환하므로, 휠 재조절 시 표기가 중첩되지 않습니다.
- **고정 태그**: 퀄리티·작가 태그를 LLM·DB를 거치지 않고 결과에 그대로 붙입니다. 퀄리티·작가 각각 **맨앞/맨뒤 위치**를 독립으로 고르고, 둘이 같은 위치면 **우선순위**(퀄리티/작가 먼저)로 순서를 정합니다. 출력 순서는 `[앞 고정] + [패스스루 + Danbooru 태그] + [뒤 고정]`입니다. 퀄리티·작가 태그 모두 뱃지로 표시되며 휠로 가중치를 조절합니다. 작가 태그는 `없음/@/artist:` 접두사를 지원합니다. 고정 태그는 언더스코어 치환·괄호 이스케이프를 받지 않습니다.
- **결과 태그 직접 편집**: Danbooru 태그를 textarea에서 직접 수정할 수 있습니다(배지와 양방향 동기화). DB 태그 배지는 마우스 휠로 가중치(0.1 단위)를 조절합니다. 1.0이면 태그만, 그 외엔 현재 표기(로컬/NAI)로 출력됩니다.

### 로그

- **콘솔 로그**: 요청이 들어오면 수신(📥)·대기 인원, 생성마다 모드(🟢 일반 / 🔵 고급)·사용자 입력(📝)·설정(⚙️ 톤/검색범위/온도)·최종 태그(✨), 처리 실패 시 에러(❌)를 실시간 출력합니다(`docker compose logs -f`).
- **파일 로그**: 사용자 입력 / 최종 태그(언더스코어→공백) / 자연어를 JSON Lines로 일자별 저장합니다(`logs/`). 정상 완료 시에만 기록합니다.

### 인덱스 variant

- **nt** (본선, gemma2 프로필): 위 "임베딩" 참고. 경로는 `data/embeddinggemma-2/lancedb_nt`입니다.

e5 프로필(되돌리기용)의 인덱스. 실측 결과 **B를 본선으로 확정**했었습니다.

- **b** (e5 본선): 영어 태그명 + 한국어 별칭. 실사용 입력의 한·영 코드스위칭 비율이 높아 영어 태그명 포함이 유리합니다.
- **c** (선택): 한국어 정의 + 별칭. 순수 한국어 의미검색용으로 남겨뒀습니다.
- ~~a~~: 폐기했습니다(전부 포함했더니 신호가 평균으로 뭉개졌습니다).

### 준비

1. 의존성 설치: `uv sync` 또는 `pip install -e .`
2. `.env` 작성 (LLM 엔드포인트, `EMBEDDING_DEVICE`)
3. `danbooru-tags.csv`를 루트에 배치 (`name,category,post_count,description`)
4. 임베딩 모델 다운로드(최초 1회): `python download_model.py` (gemma2 프로필이면 `models/embeddinggemma-2`)
5. 인덱스 빌드: `python -m core.builder` (GPU 권장 — CPU는 매우 느립니다). gemma2 프로필은 처음 실행할 때 새 한국어 소스 CSV를 HF에서 받습니다.

> Python 3.11 이상, sentence-transformers 6.1 이상, transformers 5.22 이상이 필요합니다(embeddinggemma-2).

> torch는 GPU 환경에 맞는 CUDA 빌드를 별도로 설치해야 합니다(예: cu130). `lancedb` 외에 `pylance`(lance 바인딩)와 `pandas`가 필요합니다(pyproject에 명시돼 있습니다).

### 사용

- API 서버: `python -m core.api` (포트 3333)
- 브라우저: `http://localhost:3333`
- 단일 쿼리 직접검색·디버깅: UI 우측 패널 또는 `/api/direct_search`

### 배포

Docker 기반. 자세한 절차는 [DEPLOY.md](DEPLOY.md) 참고.

```bash
docker compose up -d --build
curl http://localhost:3333/api/health
```

모델·LanceDB·CSV·.env는 이미지에 굽지 않고 볼륨 마운트합니다. 인덱스는 GPU 머신에서 빌드해 복사합니다(임베딩 벡터는 디바이스와 무관합니다). `index.html`은 이미지에 COPY되므로 UI 수정 후 `--build` 재빌드가 필요합니다.

### 구조

| 파일 | 역할 |
|------|------|
| `config.py` | 전역 설정 (`.env` 로드) |
| `parser.py` | description 타입별 분기 파싱 + variant별 임베딩 텍스트 구성 |
| `embeddings.py` | 임베딩 (gemma2: embeddinggemma-2 / e5: e5-large) |
| `database.py` | LanceDB (카테고리 필터 검색 지원) |
| `builder.py` | CSV → variant 인덱스 빌드 (nt 는 `kr2.py` 로 새 한국어 소스 적용) |
| `kr2.py` | 새 한국어 소스(mengod-hf/danbooru-tags-kr) 로드·교체·nt 문서 텍스트 |
| `search.py` | 벡터검색 + 2분할(일반/캐릭터) + 환각필터용 태그집합 + 이름 오염차단 헬퍼 + v3용 사전필터·어휘일치 |
| `llm.py` | LLM 호출 (분해 선별·캐릭터 선택·자연어·톤, v3용 JSON 스키마 호출 `_call_json`) |
| `llm_v3.py` | v3 구조화 분해 / 조각별 선택 프롬프트 |
| `pipeline_v3.py` | **본선 파이프라인** (gemma2 프로필 기본) |
| `llm_decompose.py` | 한국어 분해 / 통번역 분해 / 태그 완성 프롬프트 |
| `pipeline_decomposed.py` | 기존 4회 분해 파이프라인 (`PIPELINE=decomposed`, e5 프로필 기본) |
| `pipeline_split.py` | 고급(분할) 모드 파이프라인 (인물칸/배경칸) |
| `pipeline.py` | 구버전 2-pass (벤치마크 비교 전용, 본선 아님) |
| `genlog.py` | 생성 로그 (콘솔 + 파일 JSONL) |
| `api.py` | FastAPI + 정적서빙 + 대기열 |
| `benchmark*.py` | 설계 검증용 측정 스크립트 |

설계 결정은 진단셋이 아니라 실사용 로그 분석과 benchmark 측정으로 내렸습니다. 측정 스크립트를 함께 둔 이유입니다.

---

## English

### What it solves

Translating Korean to English tags with an LLM breaks two ways:

1. **Hallucination** — invents tags not in the DB (e.g. "은발" → `silver_hair`, which doesn't exist; Danbooru merges it into `grey_hair`).
2. **Mode collapse** — flattens fine detail ("dragon horns on both sides of head") into head terms.

Both are blocked structurally.

- **Hallucination-free**: final tags pass a triple filter — LLM prompt constraint, candidate-pool check, and a DB-existence code filter. Whatever the LLM emits, anything absent from the DB is dropped by code.
- **Detail preservation**: Korean is decomposed into semantic search units, each retrieved separately; the LLM selects from real candidates against the original intent.

### Flow (basic mode, v3 — since 2026-11-05)

1. Structured decomposition (JSON-schema constrained) → people / characters / units `{ko, en}` (one concept per unit)
2. Per-unit candidates → exact tag-name match + Korean query top5 + English query top5 (category `where` prefilter)
3. Per-unit selection by candidate number only → a short English phrase only when no candidate matches
4. Final = headcount tags + character tags + selected tags/phrases (+ DB-existence and cosplay double guard)
5. Natural-language prompt (tones unchanged)

Embedding is `google/embeddinggemma-2` (text encoder only, 768-d) with index `nt` (definitions/aliases from the `mengod-hf/danbooru-tags-kr` source, embedded in the model card's `title: <tag> | text: <definition aliases>` format). On a 10-sentence benchmark with the same LLM (gemma4 26B A4B), required-concept coverage went from 82% (previous 4-call pipeline) to 95%, and two runs were 10/10 identical. Set `EMBEDDING_PROFILE=e5` to return to e5-large and the previous pipeline below (`PIPELINE=decomposed` switches only the pipeline).

Previous 4-call pipeline (`PIPELINE=decomposed`, default for the e5 profile):

0. Character resolution → extract (series?, character) pairs → cat3/4 retrieval → batched selection (skipped when no character is present)
1. Decompose Korean + retrieve → Korean-unit candidates
2. Translate-whole + decompose + retrieve → English-unit candidates
3. Complete tags → select against original + hallucination code-filter + headcount protection
4. Generate natural-language prompt

If a sentence names a character (e.g. "Raiden Shogun from Genshin in the rain"), it is resolved automatically without separate fields: the LLM extracts (series?, character) pairs, retrieves character/series candidates, and picks per pair in one call. Confirmed character tags lead the final output; series tags feed NL only; the extracted names are stripped from general search to avoid `_(cosplay)` pollution.

Key structures: a **two-way DB split** (general vs character/series) to stop character tags from polluting attribute search; **headcount protection** so multi-person scenes aren't collapsed to solo; **step streaming** with a single-flight queue and live wait position.

### Two input modes

**Basic mode**: one sentence through the pipeline. A **tag search scope** selector (general/artist/copyright/character/meta) controls which categories are searched; default is general, and enabling non-general categories can degrade output quality.

**Advanced split mode**: per-character fields + background. Each character picks a **source** of three: *DB lookup* (LLM picks character/series tags from DB candidates by user intent; series tags feed inference/NL only, not the final output), *unlisted character (passthrough)* (skips DB search, uses the typed name verbatim in NL and at the front of the tags), or *original* (skips search, NL invents a name). Character names in the description/background fields are stripped to avoid `_(cosplay)` pollution.

**Natural-language tone** presets control verbosity, with temperature auto-set per tone. *Descriptive* writes two or more flowing sentences; *plain* gives short, modifier-free sentences and is fed the pre-convergence English decomposition units (not the final tags) to recover meaning lost during tag convergence, while preserving character identity; *phrase+* emits a comma-separated list of short English phrases instead of sentences — an alternative for when longer prose destabilizes the art style — carrying only the residual (visual detail and explicitly-stated mood) the tags don't express, with each tag's covered meaning given to the LLM as an exclusion context. When a character tag is present, NL opens with `<Character> from <Series>, ...`; the source structure is preserved regardless of tone.

**Weight syntax (local / NAI)**: all weights render as local `(tag:weight)` or NAI `weight::tag::`; NAI supports negative weights (floor -10). State is always stored in local form and converted only on display/copy, so repeated wheel adjustments never nest the notation.

**Fixed tags** (quality and artists) bypass the LLM/DB and are appended verbatim. Quality and artists each independently choose a **front/back position**; when both share a side, a **priority** toggle (quality/artist first) sets the order. Output order is `[front fixed] + [passthrough + Danbooru tags] + [back fixed]`. Both quality and artist tags render as badges with mouse-wheel weight adjustment; artists support a `none/@/artist:` prefix. Result tags are editable in a textarea (two-way synced with badges).

### Index variant

With the default gemma2 profile the mainline index is **nt** (`data/embeddinggemma-2/lancedb_nt`). For the e5 profile (rollback), variant **b** (English tag name + Korean aliases) was confirmed as the mainline from measurements — real-world input is heavily code-switched, so including English tag names helps. Variant **c** kept as an option; **a** dropped.

### Setup / Run

```bash
uv sync                      # or pip install -e .
# .env, danbooru-tags.csv at root
python download_model.py     # once
python -m core.builder       # GPU recommended
python -m core.api           # port 3333
```

torch must be installed as a CUDA build matching your GPU. `pylance` (lance binding) and `pandas` are required alongside `lancedb`.

### Deploy

Docker-based; see [DEPLOY.md](DEPLOY.md). Models, LanceDB, CSV and `.env` are volume-mounted, not baked into the image. Build indexes on a GPU machine and copy them over. `index.html` is COPYed into the image, so rebuild with `--build` after UI changes.

### License

MIT
