# 배포 가이드 (4060 서버, 포트 3333)

## 사전 준비 (호스트)

배포 디렉토리에 아래가 있어야 합니다 (이미지에 굽지 않고 마운트합니다):

```
danbooru-tags-rag/
├── data/                     # LanceDB 인덱스 — 5070에서 빌드해 복사 (variant b 필수)
│   ├── lancedb_b/            # ★ 본선. 없으면 검색 전부 실패
│   ├── lancedb_c/            # 선택 (직접조회 비교용)
│   └── lancedb_a/            # 폐기 — 안 올려도 됨
├── models/
│   └── multilingual-e5-large/   # e5 모델 (download_model.py 산출물)
├── danbooru-tags.csv         # 태그 호버 설명용 (UI가 /danbooru-tags.csv 로 fetch)
├── .env                      # 아래 참고
├── logs/                     # 생성 로그 저장 위치 (쓰기 가능, 자동 생성되나 미리 mkdir 권장)
├── index.html                # UI (이미지에 COPY됨 — 수정 시 --build 필요)
├── core/ Dockerfile docker-compose.yml pyproject.toml
```

빌드 산출물(data/)은 **5070 Ti에서 빌드**해서 복사합니다. 4060에서 CPU 빌드는 금지입니다(2시간 소요).
임베딩 벡터는 device와 무관하게 동일하므로 cuda로 빌드한 인덱스를 cpu 서빙에 그대로 씁니다.

## .env

```dotenv
# 컨테이너 → 호스트 llama.cpp. host.docker.internal 게이트웨이 사용.
LLM_API_URL=http://host.docker.internal:8080/v1/chat/completions
LLM_MODEL=gemma4
LLM_THINKING=off
EMBEDDING_DEVICE=cpu
```

llama.cpp 가 4060 호스트에서 8080 등으로 떠 있어야 합니다.
포트가 다르면 LLM_API_URL 을 수정합니다.

## 실행

```bash
mkdir -p logs                   # 생성 로그 디렉토리(없으면 docker가 root 소유로 자동생성됨)
docker compose up -d --build
docker compose logs -f          # "준비 완료" 뜨면 OK
```

확인:

```bash
curl http://localhost:3333/api/health      # {"status":"ok","variant":"b"}
```

브라우저: http://<서버>:3333

> UI(index.html)를 수정한 경우 이미지에 COPY되므로 **반드시 `--build`로 재빌드**해야 반영됩니다.

## 점검 포인트 (배포 직후)

- /api/health → variant b 확인
- UI에서 한국어 묘사 → Danbooru태그 + 자연어 정상 출력
- 캐릭터 이름 포함 입력(예: "원신 라이덴 쇼군") → 캐릭터 태그가 맨 앞에 오는지 확인
- 태그 칩에 마우스 올리면 CSV 설명 툴팁(없으면 CSV 서빙/경로 확인)
- 직접조회(우측)로 "은발" 검색 → grey_hair 등 일반태그 확인
- /api/logs 로 환각제거·폴백 빈도 모니터링

## 클라이언트 캐시 (업데이트 반영)

`index.html`은 `Cache-Control: no-cache, no-store, must-revalidate`로 서빙되어, 업데이트 후
일반 새로고침만으로 최신 UI가 반영됩니다(강제 새로고침 불필요). `danbooru-tags.csv`는 `no-cache`로
ETag 재검증(미변경 시 304)하여 묵은 파일을 막으면서 대역폭을 절약합니다.

> 단, 캐시 헤더가 박히기 **이전** 버전을 캐싱 중인 기존 사용자는 이번 배포 한 번만 강제 새로고침이
> 필요할 수 있습니다. 그 이후부터는 자동입니다. Cloudflare에 "Cache Everything" 규칙을 걸어둔
> 경우 HTML이 엣지 캐시될 수 있으니 `/` 경로를 캐시 예외로 둡니다.

## 생성 로그

### 콘솔 로그

`docker compose logs -f` 로 실시간 확인.

- 📥 요청 수신(모드 + 큐 N명 대기)
- 🟢 일반 / 🔵 고급 모드 · 📝 사용자 입력 · ⚙️ 설정(톤/검색범위/온도) · ✨ 최종 태그 · 자연어
- ❌ 생성 실패(파이프라인 예외/연결끊김 — 콘솔에만, 파일 미기록)
- 🔍 파이프라인 검색 / 🔎 직접조회

### 파일 로그 (저장)

생성이 **정상 완료**되면 사용자 입력 / 최종 태그 / 최종 자연어가 파일에 한 줄씩(JSON Lines) 남습니다.
기본 모드(`/api/generate`)·고급 분할 모드(`/api/generate_split`) 둘 다 기록됩니다.
(실패·연결끊김 요청은 콘솔에만 남고 파일엔 기록되지 않습니다.)

- 위치: `logs/generations-YYYY-MM-DD.jsonl` (일자별 분리 — 무한 증식 방지).
  - compose 의 `./logs:/app/logs` 볼륨으로 **호스트에 보존**(컨테이너 재시작/재빌드해도 유지).
  - 경로는 `GEN_LOG_PATH` 환경변수로 변경 가능(기본 `/app/logs/generations.jsonl`).
- 한 줄 포맷:
  ```json
  {"ts":"...", "mode":"basic|split", "input":"사용자 입력",
   "final_tags":["1girl","purple hair",...],
   "final_tags_str":"1girl, purple hair, ...",
   "nl_prompt":"최종 자연어", "extra":{...분할모드 원본 구조...}}
  ```
- **최종 태그는 언더스코어(`_`)를 공백으로 치환해서 저장합니다**(`purple_hair` → `purple hair`).
  `final_tags`(배열)·`final_tags_str`(한 줄) 둘 다 치환본. 원본 언더스코어 형태는 별도 보존하지 않습니다.
- 빠른 확인:
  ```bash
  tail -f logs/generations-$(date +%F).jsonl
  # 사람이 읽기 좋게:
  tail -n 20 logs/generations-$(date +%F).jsonl | jq -r '"\(.ts) [\(.mode)] \(.input)\n  → \(.final_tags_str)\n  → \(.nl_prompt)\n"'
  ```
- 주의: 단일 워커 전제(append 경합 방지). 워커를 늘리지 마십시오. `.gitignore` 에 `logs/` 가 포함되어
  커밋에는 섞이지 않습니다. 기록 실패는 응답을 막지 않습니다(genlog 가 예외를 삼킵니다) — 로그가 안
  남으면 디렉토리 쓰기 권한부터 확인합니다.

## 알려진 미확정 (실데이터로 보정 예정)

- **별칭매칭(bigram) 미구현** — "은발→grey_hair"가 top_k 밖이면 못 잡는 케이스 잔존(최우선 과제).
- **일반 모드 캐릭터 해석** — 신규(2단계: 쌍 추출 → 일괄 판별). 실데이터로 검증 필요:
  (a) 일반인("여자")을 캐릭터로 오추출하는지, (b) 작품명 없는 동명이인 판별 정확도,
  (c) LLM 호출 2회 추가에 따른 큐 대기 체감.
- **인원수 정렬** — 프롬프트 규칙(RULE 5)으로 1차 강제. 어기는 빈도 잦으면 코드 후처리 검토.
- **캐릭터 오프닝 / 출처구조 보존** — 프롬프트 규칙이라 100% 보장 아님. 실배포 NL 로그로 확인.
- **캐릭터 폴백 threshold** — 현재 DEFAULT_THRESHOLD(0.80). cross-lingual score 압축이라 보정 대상.

## 주의

- uvicorn 단일 프로세스(워커 1)입니다. 늘리면 인메모리 로그/큐가 프로세스별로 갈립니다.
- reload 를 켜지 마십시오(좀비 프로세스 + 멀티프로세스 로그 분산).
- CORS allow_origins 가 현재 "\*" 입니다. 같은 오리진 서빙이라 무방하나, 외부 노출 시 도메인으로 좁힙니다.
- 작가(cat1)·메타(cat5)는 인덱싱하도록 바뀌었으나(EXCLUDED_CATEGORIES 비움), **인덱스 재빌드를 해야
  실제로 검색됩니다.** 재빌드 전에는 태그 검색 범위/직접조회에서 작가·메타를 선택해도 0건입니다.
  5070 Ti에서 빌드해 4060으로 복사합니다(CPU 빌드 금지).

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
