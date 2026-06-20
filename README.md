# danbooru-tags-rag

한국어 입력을 받아 cross-lingual 검색으로 Danbooru 태그와 자연어 프롬프트를 생성하는 로컬 도구.
로컬 임베딩(multilingual-e5-large) + 로컬 LLM(llama.cpp) 기반. 환각 없이 동작한다.

A local Korean→Danbooru tag generator using cross-lingual retrieval over local embeddings
(multilingual-e5-large) and a local LLM (llama.cpp). Hallucination-free by construction.

---

## 한국어

### 풀려는 문제

언어모델로 한국어를 영어 태그로 바꾸면 두 가지가 깨진다.

1. **환각** — DB에 없는 태그를 지어낸다. (예: 은발 → `silver_hair`. Danbooru엔 그 태그가 없고 `grey_hair`로 통합돼 있다.)
2. **세부 표현 붕괴(mode collapse)** — "머리 양쪽 드래곤 뿔" 같은 디테일을 head 단어로 뭉갠다.

이 도구는 둘 다 **구조적으로** 막는다.

- **환각 차단**: 최종 태그는 (1) LLM 프롬프트로 "후보 외 금지" 강제 + (2) 후보풀 대조 + (3) DB 실존 태그 코드필터, 삼중으로 거른다. LLM이 무엇을 생성하든 DB에 없으면 코드가 버린다.
- **세부 보존**: 단순 번역이 아니라 한국어 의미단위로 분해해 각각 검색하고, 실존 후보 중에서 원본 의도에 맞는 것을 LLM이 선택한다.

### 동작 흐름 (4-step)

```
입력(한국어)
 1. 한국어 분해 + 검색      → 한국어 단위별 후보
 2. 통번역 분해 + 검색      → 영어 단위별 후보  (문맥 보존 위해 통째로 번역 후 분해)
 (한국어/영어 후보풀 합본)
 3. 태그 완성              → 원본 기준 선별 + 환각 코드필터 + 인원수 보호
 4. 자연어 프롬프트 생성
```

- **2분할 DB**: 일반(cat 0) / 캐릭터·작품(cat 3,4)을 분리 검색한다. 데이터셋이 작아 캐릭터 태그가 속성 검색을 오염시키는 현상(은발 → 특정 캐릭터)을 카테고리 필터로 차단. 작가(1)·메타(5)는 빌드 단계에서 제외.
- **인원수 보호**: 검색으로 확정하되 LLM 선별 단계에서 인원수 슬롯을 따로 보호해, 다인물 장면이 solo로 뭉개지는 것을 막는다.
- **단계별 스트리밍**: 각 단계가 끝나는 즉시 UI에 표시(키워드 먼저 보이고 최종은 나중). 동시 요청은 1개씩 직렬 처리하며 대기 순번을 실시간 표시한다.

### 인덱스 variant

임베딩 텍스트 구성이 다른 인덱스. 실측 결과 **B를 본선으로 확정**했다.

- **b** (본선): 영어 태그명 + 한국어 별칭. 실사용 입력의 한·영 코드스위칭 비율이 높아 영어 태그명 포함이 유리.
- **c** (선택): 한국어 정의 + 별칭. 순수 한국어 의미검색용으로 남겨둠.
- ~~a~~: 폐기 (전부 포함했더니 신호가 평균으로 뭉개짐).

### 준비

1. 의존성 설치: `uv sync` 또는 `pip install -e .`
2. `.env` 작성 (LLM 엔드포인트, `EMBEDDING_DEVICE`)
3. `danbooru-tags.csv` 를 루트에 배치 (`name,category,post_count,description`)
4. 임베딩 모델 다운로드(최초 1회): `python download_model.py`
5. 인덱스 빌드: `python -m core.builder` (GPU 권장 — CPU는 매우 느림)

> torch는 GPU 환경에 맞는 CUDA 빌드를 별도 설치해야 한다(예: cu130). `lancedb` 외에 `pylance`(lance 바인딩)와 `pandas`가 필요하다(pyproject에 명시됨).

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

모델·LanceDB·CSV·.env는 이미지에 굽지 않고 볼륨 마운트한다. 인덱스는 GPU 머신에서 빌드해 복사한다(임베딩 벡터는 디바이스 무관).

### 구조

| 파일 | 역할 |
|------|------|
| `config.py` | 전역 설정 (`.env` 로드) |
| `parser.py` | description 타입별 분기 파싱 |
| `embeddings.py` | e5-large 임베딩 |
| `database.py` | LanceDB (카테고리 필터 검색 지원) |
| `builder.py` | CSV → variant 인덱스 빌드 |
| `search.py` | 벡터검색 + 2분할(일반/캐릭터) + 환각필터용 태그집합 |
| `llm.py` | LLM 호출 (키워드/정제/선택/자연어) |
| `llm_decompose.py` | 한국어 분해 / 통번역 분해 / 태그 완성 프롬프트 |
| `pipeline_decomposed.py` | **4-step 본선 파이프라인** (+ 스트리밍 제너레이터) |
| `pipeline.py` | 구버전 2-pass (측정 비교용, 본선 아님) |
| `api.py` | FastAPI + 정적서빙 + 대기열 |
| `benchmark*.py` | 설계 검증용 측정 스크립트 (raw/pipe/분해/2분할) |

설계 결정은 진단셋이 아니라 실사용 로그 분석과 benchmark 측정으로 내렸다. 측정 스크립트를 함께 둔 이유다.

---

## English

### What it solves

Translating Korean to English tags with an LLM breaks two ways:

1. **Hallucination** — invents tags not in the DB (e.g. "은발" → `silver_hair`, which doesn't exist; Danbooru merges it into `grey_hair`).
2. **Mode collapse** — flattens fine detail ("dragon horns on both sides of head") into head terms.

Both are blocked structurally.

- **Hallucination-free**: final tags pass a triple filter — LLM prompt constraint, candidate-pool check, and a DB-existence code filter. Whatever the LLM emits, anything absent from the DB is dropped by code.
- **Detail preservation**: Korean is decomposed into semantic search units, each retrieved separately; the LLM selects from real candidates against the original intent.

### Flow (4-step)

1. Decompose Korean + retrieve → Korean-unit candidates
2. Translate-whole + decompose + retrieve → English-unit candidates
3. Complete tags → select against original + hallucination code-filter + headcount protection
4. Generate natural-language prompt

Key structures: a **two-way DB split** (general vs character/series) to stop character tags from polluting attribute search; **headcount protection** so multi-person scenes aren't collapsed to solo; **step streaming** with a single-flight queue and live wait position.

### Index variant

Variant **b** (English tag name + Korean aliases) is confirmed as the mainline from measurements — real-world input is heavily code-switched, so including English tag names helps. Variant **c** kept as an option; **a** dropped.

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

Docker-based; see [DEPLOY.md](DEPLOY.md). Models, LanceDB, CSV and `.env` are volume-mounted, not baked into the image. Build indexes on a GPU machine and copy them over.

### License

MIT
