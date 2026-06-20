# danbooru-tags-rag

한국어 입력을 받아 2-pass cross-lingual 검색으로 Danbooru 태그와 자연어 프롬프트를 생성하는 로컬 도구.
로컬 임베딩(multilingual-e5-large) + 로컬 LLM 기반.

A local tool that takes Korean input and produces Danbooru tags and a natural-language prompt
via 2-pass cross-lingual retrieval. Built on local embeddings (multilingual-e5-large) and a local LLM.

---

## 한국어

### 개요

언어모델은 한국어를 영어 태그로 번역할 때 세부 표현을 대중적인 단어로 뭉개는(mode collapse) 경향이 있고,
존재하지 않는 태그를 지어내는 환각도 발생한다. 이 도구는 두 문제를 구조적으로 해결한다.

- **환각 차단**: 모든 태그는 벡터 DB를 통과해서만 나오므로, DB에 없는 태그는 구조적으로 생성 불가능.
- **세부 표현 보존**: 단순 번역이 아니라 "DB가 떠준 실존 후보를 LLM이 보고 의도에 맞게 선택"하는
  2-pass 방식으로, head 단어로 수렴하는 현상을 완화.

### 동작 흐름 (2-pass)

1. **Pass 1** — 한국어 입력을 거친 그물로 직접 검색 → 후보 태그
2. **Refine** — 후보를 LLM에 보여주고 영어 의도표현으로 정제
3. **Pass 2** — 정제된 영어로 정밀 검색
4. **Select** — 카테고리별로 구조화된 실존 후보 중 LLM이 최종 선택
5. **NL/Verify** — 자연어 프롬프트 생성 + 의심 태그 표시(사용자 확인용)

### 인덱스 variant (a/b/c)

임베딩 텍스트 구성이 다른 3개 인덱스를 만들어 비교한다.

- **a**: 영어 태그명 + 한국어 정의 + 별칭 (전부)
- **b**: 영어 태그명 + 한국어 별칭 (정의 제외)
- **c**: 한국어 정의 + 별칭 (영어 태그명 제외)

### 준비

1. 의존성 설치: `pip install -e .`
2. `.env` 작성 (`.env.example` 복사 후 값 채우기)
3. `danbooru-tags.csv` 를 루트에 배치 (`name,category,post_count,description`)
4. 임베딩 모델 다운로드(최초 1회): `python download_model.py`
5. 인덱스 빌드: `python -m core.builder`

### 사용

- API 서버: `python -m core.api`
- variant 비교: `python -m core.benchmark raw` (순수 검색) / `pipe` (파이프라인)

### 구조

| 파일 | 역할 |
|------|------|
| `config.py` | 전역 설정 (`.env` 로드) |
| `parser.py` | description 타입별 분기 파싱 |
| `embeddings.py` | e5-large 임베딩 (GPU) |
| `database.py` | LanceDB (variant a/b/c) |
| `builder.py` | CSV → 3-variant 인덱스 빌드 |
| `search.py` | 순수 벡터검색 빌딩블록 |
| `llm.py` | LLM 호출 함수 |
| `pipeline.py` | 2-pass 오케스트레이션 |
| `api.py` | FastAPI (얇은 계층) |
| `benchmark.py` | variant 비교 도구 |

---

## English

### Overview

Language models tend to collapse fine-grained Korean expressions into common English words
(mode collapse) when translating to tags, and may hallucinate non-existent tags.
This tool addresses both structurally.

- **Hallucination-free**: every tag must pass through the vector DB, so tags absent from the DB
  are structurally impossible to produce.
- **Detail preservation**: instead of plain translation, the LLM *selects* from real candidates
  retrieved by the DB (2-pass), mitigating collapse toward head terms.

### Pipeline (2-pass)

1. **Pass 1** — coarse direct search on the Korean input → candidate tags
2. **Refine** — show candidates to the LLM, refine into English intent expressions
3. **Pass 2** — precise search with the refined English
4. **Select** — LLM picks final tags from category-structured real candidates
5. **NL/Verify** — generate a natural-language prompt + flag suspicious tags (for user review)

### Index variants (a/b/c)

Three indexes with different embedding-text composition, for comparison.

- **a**: English tag name + Korean definition + aliases (all)
- **b**: English tag name + Korean aliases (no definition)
- **c**: Korean definition + aliases (no English tag name)

### Setup

1. Install: `pip install -e .`
2. Create `.env` (copy from `.env.example`)
3. Place `danbooru-tags.csv` at the root (`name,category,post_count,description`)
4. Download the embedding model (once): `python download_model.py`
5. Build indexes: `python -m core.builder`

### Usage

- API server: `python -m core.api`
- Compare variants: `python -m core.benchmark raw` (raw search) / `pipe` (pipeline)

### Layout

| File | Role |
|------|------|
| `config.py` | Global config (loads `.env`) |
| `parser.py` | Per-type description parsing |
| `embeddings.py` | e5-large embedding (GPU) |
| `database.py` | LanceDB (variant a/b/c) |
| `builder.py` | CSV → 3-variant index build |
| `search.py` | Pure vector-search building blocks |
| `llm.py` | LLM call functions |
| `pipeline.py` | 2-pass orchestration |
| `api.py` | FastAPI (thin layer) |
| `benchmark.py` | Variant comparison tool |

---

## License

MIT © joykst96
