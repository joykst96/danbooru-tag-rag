# danbooru-tags-rag

한국어 묘사를 받아 Danbooru 태그와 자연어 프롬프트를 만드는 로컬 도구입니다.
로컬 임베딩(`google/embeddinggemma-2`)과 로컬 LLM(llama.cpp)으로 동작하고, DB에 없는 태그는 내보내지 않습니다.

A local Korean→Danbooru tag generator built on a local embedding model (`google/embeddinggemma-2`)
and a local LLM (llama.cpp). Tags that are not in the database never reach the output.

---

## 한국어

### 무엇을 해결하나

언어모델로 한국어를 영어 태그로 바꾸면 두 가지가 깨집니다.

1. **환각**: DB에 없는 태그를 지어냅니다. (예: 은발 → `silver_hair`. Danbooru에는 그 태그가 없고 `grey_hair`로 통합돼 있습니다.)
2. **세부 표현 붕괴**: "머리 양쪽 드래곤 뿔" 같은 디테일을 큰 단어 하나로 뭉갭니다.

이 도구는 둘 다 구조로 막습니다. 문장을 개념 하나짜리 조각으로 나눠 조각마다 실제 태그 후보를 검색하고, LLM은 그 후보 중에서 **번호로만** 고릅니다. 맞는 후보가 없는 조각은 버리지 않고 짧은 영어 구로 남깁니다.

```
한국어 입력 → 조각 분해 → 조각별 태그 후보 검색 → 후보 중 선택 → 태그 + 자연어 프롬프트
```

### 기능

- **기본 모드**: 한 문장을 통째로 넣습니다.
- **고급(분할) 모드**: 인물칸 여러 개와 배경칸 하나로 나눠 넣습니다. 다인물 장면에서 인물이 섞이지 않습니다. 인물칸마다 캐릭터를 DB에서 찾을지, DB에 없는 이름을 그대로 쓸지(패스스루), 오리지널로 둘지 고릅니다.
- **자연어 톤**: 묘사적 / 담백 / 단어형+(짧은 영어 구 나열) 중에서 고릅니다.
- **출력 편집**: 가중치 표기(로컬 `(tag:1.2)` / NAI `1.2::tag::`), 퀄리티·작가 고정 태그, 결과 태그 직접 편집, 뱃지 휠로 가중치 조절을 지원합니다.
- **직접 검색**: 단어 하나로 태그 DB를 바로 찾아봅니다(UI 우측 패널, `/api/direct_search`).

### 요구 사항

- Python 3.11 이상
- `response_format: json_schema`(문자열 `pattern` 포함)를 지원하는 OpenAI 호환 LLM 서버(예: llama.cpp)
- 인덱스 빌드용 GPU(권장). 서빙은 CPU로 됩니다.

### 설치와 실행

```bash
uv sync                      # 또는 pip install -e .   (torch 는 환경에 맞는 CUDA 빌드를 따로 설치)
cp .env.example .env         # LLM 엔드포인트, EMBEDDING_DEVICE 설정
python download_model.py     # 최초 1회 → models/embeddinggemma-2
python -m core.tags_source   # 태그 CSV → danbooru-tags.csv
python -m core.builder       # 인덱스 빌드 → data/embeddinggemma-2/lancedb_sim (GPU 권장)
python -m core.api           # http://localhost:3333
```

### 태그 데이터

태그와 한국어 설명은 Hugging Face 데이터셋 [mengod-hf/danbooru-tags-kr](https://huggingface.co/datasets/mengod-hf/danbooru-tags-kr)을 씁니다.
`python -m core.tags_source`가 고정한 revision을 받아 이 도구의 형식으로 바꾼 `danbooru-tags.csv`를 만듭니다. 이미지 50장 이상인 태그만 남기고, 별칭 표기를 정리합니다. 이 CSV는 저장소에 들어 있지 않습니다.

데이터는 이 저장소의 MIT 라이선스가 아니라 **데이터셋의 라이선스**를 따릅니다. 한국어 번역·분류·설명은 CC BY-SA 4.0이고, 일부 상위 자료는 각자의 조건(MIT 고지 등)을 따릅니다. 만든 CSV를 배포하거나 서비스에서 내려 줄 때는 출처와 변경 사항을 표시하고, 같은 라이선스로 공유해야 합니다. 자세한 내용은 데이터셋의 [LICENSE](https://huggingface.co/datasets/mengod-hf/danbooru-tags-kr/blob/main/LICENSE)를 보세요.

### 배포

Docker로 배포합니다. 절차는 [DEPLOY.md](DEPLOY.md)에 있습니다.

### 더 읽을 문서

- [DEPLOY.md](DEPLOY.md): 배포, 설정, 생성 로그, 알려진 과제
- [MAINTENANCE.md](MAINTENANCE.md): 내부 동작과 설계 이유, 코드 구조, 태그 데이터 갱신 절차

---

## English

### What it solves

Translating Korean to English tags with an LLM breaks two ways:

1. **Hallucination**: it invents tags that are not in the DB (e.g. "은발" → `silver_hair`, which doesn't exist; Danbooru merges it into `grey_hair`).
2. **Detail collapse**: it flattens fine detail ("dragon horns on both sides of the head") into a head term.

Both are blocked by construction. The sentence is split into one-concept units, real tag candidates are retrieved per unit, and the LLM may only pick candidates **by number**. A unit with no matching candidate is kept as a short English phrase instead of being dropped.

### Features

- **Basic mode**: one sentence in.
- **Advanced split mode**: per-character fields plus one background field, so characters don't mix in multi-person scenes. Each character is looked up in the DB, passed through verbatim, or marked original.
- **Natural-language tone**: descriptive / plain / phrase+ (short comma-separated phrases).
- **Output editing**: local `(tag:1.2)` or NAI `1.2::tag::` weight syntax, fixed quality/artist tags, editable result tags, wheel-adjustable weights.
- **Direct search**: look up the tag DB with a single word (right panel, `/api/direct_search`).

### Requirements

- Python 3.11+
- An OpenAI-compatible LLM server supporting `response_format: json_schema` including string `pattern` (e.g. llama.cpp)
- A GPU is recommended for building the index; serving runs on CPU.

### Setup / Run

```bash
uv sync                      # or pip install -e .   (install a CUDA build of torch for your GPU)
cp .env.example .env         # LLM endpoint, EMBEDDING_DEVICE
python download_model.py     # once → models/embeddinggemma-2
python -m core.tags_source   # tag CSV → danbooru-tags.csv
python -m core.builder       # index → data/embeddinggemma-2/lancedb_sim (GPU recommended)
python -m core.api           # http://localhost:3333
```

### Tag data

Tags and Korean descriptions come from the Hugging Face dataset [mengod-hf/danbooru-tags-kr](https://huggingface.co/datasets/mengod-hf/danbooru-tags-kr).
`python -m core.tags_source` downloads a pinned revision and converts it into this tool's `danbooru-tags.csv` (tags with 50+ posts, alias notation normalized). The CSV is not included in this repository.

The data is **not** covered by this repository's MIT license; it follows the dataset's license. The Korean translations, classifications and descriptions are CC BY-SA 4.0, and some upstream material carries its own terms (e.g. MIT notices). If you redistribute the generated CSV or serve it from a deployment, give attribution, indicate changes and share it under the same license. See the dataset's [LICENSE](https://huggingface.co/datasets/mengod-hf/danbooru-tags-kr/blob/main/LICENSE).

### Deploy

Docker-based; see [DEPLOY.md](DEPLOY.md) (Korean). Internals and data-update procedures are in [MAINTENANCE.md](MAINTENANCE.md) (Korean).

---

## Thanks

- [mengod-hf/danbooru-tags-kr](https://huggingface.co/datasets/mengod-hf/danbooru-tags-kr): 한국어 태그 번역·분류·설명 데이터 / Korean tag translations, classifications and descriptions. 이 데이터셋이 기반한 기여자들께도 감사드립니다 / with thanks to the contributors it builds on: [nyantong](https://github.com/nyantong), [PYU224/tagdb-updater](https://github.com/PYU224/tagdb-updater), [boorutan/booru-japanese-tag](https://github.com/boorutan/booru-japanese-tag), Danbooru wiki contributors.
- [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2): 임베딩 모델 / embedding model (Apache 2.0)
- [llama.cpp](https://github.com/ggml-org/llama.cpp): 로컬 LLM 서버 / local LLM server
- [Danbooru](https://danbooru.donmai.us/): 태그 체계 / the tag vocabulary

## License

코드는 MIT 라이선스입니다([LICENSE](LICENSE)). 태그 데이터와 모델은 각자의 라이선스를 따릅니다(위 "태그 데이터" 참고).

The code is MIT-licensed ([LICENSE](LICENSE)). Tag data and models follow their own licenses (see "Tag data" above).
