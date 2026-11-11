# 배포 가이드 (Docker, 기본 포트 3333)

## 사전 준비 (호스트)

배포 디렉토리에 아래가 있어야 합니다 (이미지에 굽지 않고 마운트합니다):

```
danbooru-tags-rag/
├── data/                     # LanceDB 인덱스 — GPU 머신에서 빌드해 복사
│   └── embeddinggemma-2/
│       └── lancedb_sim/      # ★ 본선 인덱스. 없으면 검색 전부 실패
├── models/
│   └── embeddinggemma-2/     # ★ 임베딩 모델 (download_model.py 산출물, 약 1.5GB)
├── danbooru-tags.csv         # python -m core.tags_source 산출물. UI 태그 설명·자동완성이 /danbooru-tags.csv 로 fetch
├── .env                      # 아래 참고
├── logs/                     # 생성 로그 저장 위치 (쓰기 가능, 자동 생성되나 미리 mkdir 권장)
├── index.html  web/          # UI (이미지에 COPY됨 — 수정 시 --build 필요)
├── core/ Dockerfile docker-compose.yml pyproject.toml
```

### 모델·인덱스 준비

```bash
python download_model.py                          # models/embeddinggemma-2 (접근 제한 없음, 약 1.5GB)
python -m core.tags_source                        # danbooru-tags.csv (HF mengod-hf/danbooru-tags-kr, post_count >= 50)
EMBEDDING_DEVICE=cuda python -m core.builder      # data/embeddinggemma-2/lancedb_sim (GPU 권장)
```

- **`danbooru-tags.csv`와 `data/embeddinggemma-2/lancedb_sim`은 항상 함께 바꿔 배포합니다.** CSV는 UI의 태그 설명·자동완성이 쓰므로, 인덱스와 같은 소스여야 설명이 맞습니다. 교체 전 파일은 백업해 두면 되돌리기 쉽습니다.
- `core.tags_source`는 소스 CSV(revision 고정)를 HF 캐시로 받습니다. 그래서 빌드 머신에는 인터넷이 필요합니다.
- 인덱스 빌드는 VRAM을 약 6GB 씁니다(토큰 예산 배치). CPU 빌드는 매우 느립니다.
- 임베딩 벡터는 디바이스와 무관하게 같으므로, CUDA로 빌드한 인덱스를 CPU 서빙에 그대로 씁니다.
- 태그 데이터를 새 판으로 갱신하는 절차는 [MAINTENANCE.md](MAINTENANCE.md) "태그 데이터 갱신"에 있습니다.

## .env

```dotenv
# 컨테이너 → 호스트 llama.cpp. host.docker.internal 게이트웨이 사용.
LLM_API_URL=http://host.docker.internal:8080/v1/chat/completions
LLM_MODEL=your-model-name
LLM_THINKING=off
EMBEDDING_DEVICE=cpu          # GPU 서빙 시 cuda
# LLM_CACHE_PROMPT=on         # 분해·선택 호출의 프롬프트 캐시 (off = 재현성 우선, 느림)
# SEARCH_BACKEND=matrix       # 후보 검색 방식. matrix = 태그 벡터를 메모리에 올려 검색(기본, 메모리 약 360MB 추가), lancedb = 예전 방식(느림)
```

파이프라인은 `response_format: json_schema`(문법 제한 디코딩)를 쓰고, 문자열에 `pattern`(대괄호·큰따옴표 금지)을 겁니다. `pattern`을 지원하지 않는 서버에서는 코드가 깨진 조각을 버리는 것으로만 방어합니다.
구조화 호출(분해·선택)은 `LLM_THINKING`과 관계없이 thinking을 끕니다. 켜면 사고 과정이 컨텍스트를 다 써서 실패할 수 있습니다.
구조화 호출은 프롬프트 캐시를 켭니다(`LLM_CACHE_PROMPT`, 기본 on). 시스템 프롬프트를 다시 처리하지 않아 빨라지며, 프롬프트 처리가 느린 서버(예: MoE 일부를 CPU에 둔 경우)일수록 효과가 큽니다.
캐시를 켜면 같은 입력의 결과가 실행마다 조금 다를 수 있습니다. 재현성이 필요하면 `LLM_CACHE_PROMPT=off`로 바꾸면 되고, 대신 느려집니다.
응답이 비거나 출력 한도까지 반복되면, 자동으로 캐시를 끄고 한 번 재시도합니다(로그: `구조화 호출 실패 → 캐시 끄고 재시도`).

호스트에 llama.cpp(또는 OpenAI 호환 chat completions 엔드포인트)가 떠 있어야 합니다.
포트·모델명이 다르면 `LLM_API_URL`·`LLM_MODEL`을 환경에 맞게 수정합니다.
컨테이너에서 호스트로 접근할 때 `host.docker.internal`이 동작하지 않는 환경(일부 Linux)에서는
compose 에 `extra_hosts: ["host.docker.internal:host-gateway"]`를 두거나 호스트 IP를 직접 지정합니다.

## 실행

```bash
mkdir -p logs                   # 생성 로그 디렉토리(없으면 docker가 root 소유로 자동생성됨)
GIT_COMMIT=$(git rev-parse --short HEAD) docker compose up -d --build   # 커밋을 생성 로그 build.commit 에 남김
docker compose logs -f          # "준비 완료" 뜨면 OK
```

`GIT_COMMIT` 없이 빌드해도 동작합니다. 그때 로그의 `build.commit`은 `unknown`이고, 코드는 `build.code_sha`(`core/*.py` 내용 해시)로 식별합니다.

확인:

```bash
curl http://localhost:3333/api/health      # {"status":"ok","variant":"sim","embedding":"embeddinggemma-2"}
```

브라우저: http://<서버>:3333

> UI(`index.html`, `web/`)를 수정한 경우 이미지에 COPY되므로 **반드시 `--build`로 재빌드**해야 반영됩니다.

## 점검 포인트 (배포 직후)

- /api/health → variant sim / embedding embeddinggemma-2 확인
- UI에서 한국어 묘사 → Danbooru태그 + 자연어 정상 출력
- 캐릭터 이름 포함 입력(예: "원신 라이덴 쇼군") → 캐릭터 태그가 맨 앞에 오는지 확인
- 태그 칩에 마우스 올리면 CSV 설명 툴팁(없으면 CSV 서빙/경로 확인)
- 직접조회(우측)로 "은발" 검색 → grey_hair 등 일반태그 확인
- 고급 모드에서 소녀 두 칸 → 최종 태그 맨 앞이 `2girls`인지 확인
- 자연어 톤 단어형+ → 결과 영역에 구가 뱃지로 뜨고 휠로 가중치 조절되는지
- 가중치 표기 NAI 토글 → 뱃지/복사가 `weight::tag::`로 바뀌고, 휠 재조절 시 표기가 중첩되지 않는지
- 고정 태그 위치(맨앞/맨뒤)·우선순위 전환 → 결과 뱃지 순서와 "모두 복사" 출력이 일치하는지
- /api/logs 로 환각제거·폴백 빈도 모니터링

## 클라이언트 캐시 (업데이트 반영)

`index.html`은 `Cache-Control: no-cache, no-store, must-revalidate`로 서빙되어, 업데이트 후
일반 새로고침만으로 최신 UI가 반영됩니다(강제 새로고침 불필요). `web/` 아래 CSS·JS와 `danbooru-tags.csv`는
`no-cache`로 ETag 재검증(미변경 시 304)하여 묵은 파일을 막으면서 대역폭을 절약합니다.

> 단, 캐시 헤더가 박히기 **이전** 버전을 캐싱 중인 기존 사용자는 이번 배포 한 번만 강제 새로고침이
> 필요할 수 있습니다. 그 이후부터는 자동입니다. Cloudflare에 "Cache Everything" 규칙을 걸어둔
> 경우 HTML이 엣지 캐시될 수 있으니 `/` 경로를 캐시 예외로 둡니다.

## 생성 로그

### 콘솔 로그

`docker compose logs -f` 로 실시간 확인.

- 📥 요청 수신(모드 + 요청 id `#xxxxxxxxxxxx` + 큐 N명 대기)
- 🟢 일반 / 🔵 고급 모드 · 📝 사용자 입력 · ⚙️ 설정(톤/검색범위/온도) · ✨ 최종 태그 · 자연어
- ❌ 생성 실패(파이프라인 예외). 파일에도 `status: "error"`로 남습니다.
- 🔍 파이프라인 검색 / 🔎 직접조회
- 콘솔의 요청 id 는 파일 로그의 `id`와 같습니다. 콘솔에서 이상한 요청을 보면 id 로 파일 레코드를 찾습니다.

### 파일 로그 (저장)

생성 요청 1건이 파일에 한 줄(JSON Lines)로 남습니다. 기본 모드(`/api/generate`)·고급 분할 모드(`/api/generate_split`) 둘 다 기록됩니다.
나중에 같은 조건으로 재현하고, 틀린 태그가 분해·검색·선택 중 어디서 생겼는지 로그만 보고 가를 수 있게 남깁니다.

- 위치: `logs/generations-YYYY-MM-DD.jsonl` (일자별 분리 — 무한 증식 방지).
  - compose 의 `./logs:/app/logs` 볼륨으로 **호스트에 보존**(컨테이너 재시작/재빌드해도 유지).
  - 경로는 `GEN_LOG_PATH` 환경변수로 변경 가능(기본 `/app/logs/generations.jsonl`).
- 형식 v2. v1 키는 이름·의미가 그대로라 기존 분석 스크립트와 아래 `jq` 명령이 그대로 동작합니다.

  | 키 | 내용 |
  |---|---|
  | `ts` | 기록 시각. v2부터 UTC 오프셋 포함(`2026-11-11T10:00:00+09:00`) |
  | `mode` | `basic` \| `split` |
  | `input` | 사용자 입력. 고급 모드는 사람이 읽기 좋은 요약(`[작품] 이름 - 묘사 / … / 배경: …`) |
  | `final_tags`, `final_tags_str` | 최종 태그, **언더스코어를 공백으로 치환**(`purple_hair` → `purple hair`) |
  | `nl_prompt` | 최종 자연어 |
  | `extra` | (고급 모드) 칸 구조. v1과 같은 형식 |
  | `v` | 형식 버전 `2`. 없으면 v1 |
  | `id` | 요청 id(12자리). 콘솔 로그와 같음 |
  | `status` | `ok` \| `error`(파이프라인 예외) \| `aborted`(연결끊김 등). v1은 `ok`만 기록했음 |
  | `final_tags_raw` | 최종 태그 원본(언더스코어 그대로) |
  | `request` | 요청 본문 전체: 톤·검색 범위·자연어 온도, 고급 모드의 `is_original`·`is_passthrough` |
  | `trace` | 단계별 중간 결과. 기본 모드: 인원수, 뺀 사람 낱말 조각, 조각마다 `ko`/`en`→후보 태그(`cands`)→선택(`picks`), 선택 메모, 환각 제거 태그. 고급 모드: 칸별 결과(이름 매칭 여부 포함), 배경 태그. 실패·중단이면 거기까지 |
  | `timing_ms` | `queue`(대기), `stages`(차례가 온 뒤 단계별 누적 ms), `total` |
  | `error` | `{type, message}` (`status: "error"`일 때) |
  | `build` | `commit`, `code_sha`, `tags_csv`(sha·행 수), `index`, `embed_model`, `llm_model`, `started`(프로세스 시작) |

  한 줄 크기는 약 2~3KB입니다(v1은 약 0.8KB). 하루 수백 건이면 하루 1MB 안팎입니다.
- 빠른 확인:
  ```bash
  tail -f logs/generations-$(date +%F).jsonl
  # 사람이 읽기 좋게:
  tail -n 20 logs/generations-$(date +%F).jsonl | jq -r '"\(.ts) [\(.mode)] \(.input)\n  → \(.final_tags_str)\n  → \(.nl_prompt)\n"'
  # 실패·중단만:
  jq -c 'select(.status != "ok") | {ts, id, status, error, input}' logs/generations-$(date +%F).jsonl
  # 한 요청의 조각별 후보·선택:
  cat logs/generations-*.jsonl | jq 'select(.id == "<요청 id>") | .trace.units[] | {ko, en, picks, cands}'
  ```
- 주의: 단일 워커 전제(append 경합 방지). 워커를 늘리지 마십시오. `.gitignore` 에 `logs/` 가 포함되어
  커밋에는 섞이지 않습니다. 기록 실패는 응답을 막지 않습니다(genlog 가 예외를 삼킵니다) — 로그가 안
  남으면 디렉토리 쓰기 권한부터 확인합니다.

## 알려진 과제

- **인원수 남은 오답**: 동물을 `others`로 세던 문제는 규칙 한 문장으로 대부분 해소했습니다(홀드아웃 24문장 한국어 21/24·영어 23/24). 남은 것은 할머니 → `others`(한국어 일부), 그림 속 인물을 세는 경우입니다.
- **캐릭터 이름으로 성별을 못 맞추는 경우**: 아냐 포저 → `1boy`.
- **근접 오답**: 창가 → `against_window`, 가장자리 → `partially_bordered` 처럼 단어는 겹치지만 뜻이 다른 태그를 고르는 경우가 남아 있습니다.
- **캐릭터 오프닝 / 출처 구조 보존**: 프롬프트 규칙이라 100% 보장은 아닙니다. 실배포 자연어 로그로 확인합니다.

## 주의

- uvicorn 단일 프로세스(워커 1)입니다. 늘리면 인메모리 로그/큐가 프로세스별로 갈립니다.
- reload 를 켜지 마십시오(좀비 프로세스 + 멀티프로세스 로그 분산).
- CORS allow_origins 가 현재 "\*" 입니다. 같은 오리진 서빙이라 무방하나, 외부 노출 시 도메인으로 좁힙니다.
- 인덱스는 모든 카테고리(작가·메타 포함)를 담습니다(`EXCLUDED_CATEGORIES` 비어 있음). 태그 검색 범위·직접조회에서 작가·메타가 0건이면 인덱스가 이 설정 이전에 빌드된 것이니 다시 빌드합니다.

## 스트리밍 / 대기열 (nginx 설정 중요)

생성 요청은 단계별로 스트리밍됩니다(캐릭터/한국어키워드→영어키워드→최종태그→자연어).
동시 요청은 1개씩 직렬 처리하며, 대기 중인 사용자에게 "앞에 N명"을 실시간 표시합니다.
(Redis 없이 프로세스 내 티켓 큐. 단일 워커 전제 — 워커 늘리면 큐가 갈라지니 1 유지.)

이게 동작하려면 리버스프록시가 응답을 버퍼링하면 안 됩니다. 두 가지로 보장합니다:

1. 앱이 `X-Accel-Buffering: no` 헤더를 보냅니다 (nginx가 이 응답만 버퍼링 해제).
2. nginx 설정에서도 명시합니다 (확실하게):

```nginx
location /api/generate {
    # prefix 매치라 /api/generate 와 /api/generate_split(분할입력) 둘 다 커버
    proxy_pass http://127.0.0.1:3333;
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 300s;     # 대기+처리 길어질 수 있음
    proxy_http_version 1.1;
}
location / {
    proxy_pass http://127.0.0.1:3333;
}
```

Cloudflare Tunnel 경유 시 Cloudflare는 ndjson 스트리밍을 대체로 통과시키지만,
혹시 버퍼링되면 위 헤더가 도움이 됩니다. 스텝이 한꺼번에 뜨면 버퍼링을 의심합니다.
