# 배포 가이드 (4060 서버, 포트 3333)

신버전(4회 분해 파이프라인 + 2분할 DB + 환각필터)을 구버전 대체로 배포한다.

## 사전 준비 (호스트)

배포 디렉토리에 아래가 있어야 한다 (이미지에 안 굽고 마운트):

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
├── index.html                # UI
├── core/ Dockerfile docker-compose.yml pyproject.toml
```

빌드 산출물(data/)은 **5070 Ti에서 빌드**해서 복사한다. 4060에서 CPU 빌드 금지(2시간).
임베딩 벡터는 device 무관하게 동일하므로 cuda로 빌드한 인덱스를 cpu 서빙에 그대로 쓴다.

## .env

```dotenv
# 컨테이너 → 호스트 llama.cpp. host.docker.internal 게이트웨이 사용.
LLM_API_URL=http://host.docker.internal:8080/v1/chat/completions
LLM_MODEL=gemma4
LLM_THINKING=off
EMBEDDING_DEVICE=cpu
```

llama.cpp 가 4060 호스트에서 8080 등으로 떠 있어야 한다(구버전과 동일).
포트가 다르면 LLM_API_URL 수정.

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

## 구버전 내리기

기존 구버전 컨테이너/프로세스를 먼저 내리고(같은 포트/리소스 충돌 방지),
Cloudflare Tunnel·nginx 가 구버전을 가리키면 신버전 3333 으로 라우팅 변경.

## 점검 포인트 (배포 직후)

- /api/health → variant b 확인
- UI에서 한국어 묘사 → Danbooru태그 + 자연어 정상 출력
- 태그 칩에 마우스 올리면 CSV 설명 툴팁(없으면 CSV 서빙/경로 확인)
- 직접조회(우측)로 "은발" 검색 → grey_hair 등 일반태그 확인
- /api/logs 로 환각제거·폴백 빈도 모니터링

## 생성 로그

### 콘솔 로그
`docker compose logs -f` 로 실시간 확인. 생성마다 모드(🟢 일반 / 🔵 고급)·📝 입력·✨ 최종 태그, 벡터DB 조회(🔍 파이프라인 / 🔎 직접조회)가 출력된다.

### 파일 로그 (저장)

생성이 정상 완료되면 **사용자 입력 / 최종 태그 / 최종 자연어**가 파일에 한 줄씩(JSON Lines) 남는다.
기본 모드(`/api/generate`)·고급 분할 모드(`/api/generate_split`) 둘 다 기록된다.

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
- **최종 태그는 언더스코어(`_`)를 공백으로 치환해서 저장**된다(`purple_hair` → `purple hair`).
  `final_tags`(배열)·`final_tags_str`(한 줄) 둘 다 치환본. 원본 언더스코어 형태가 필요하면 별도 보존 안 하므로 주의.
- 빠른 확인:
  ```bash
  tail -f logs/generations-$(date +%F).jsonl
  # 사람이 읽기 좋게:
  tail -n 20 logs/generations-$(date +%F).jsonl | jq -r '"\(.ts) [\(.mode)] \(.input)\n  → \(.final_tags_str)\n  → \(.nl_prompt)\n"'
  ```
- 주의: 단일 워커 전제(append 경합 방지). 워커 늘리지 말 것. `.gitignore` 에 `logs/` 포함되어 커밋엔 안 섞인다.
  기록 실패는 응답을 막지 않는다(genlog 가 예외를 삼킴) — 로그가 안 남으면 디렉토리 쓰기 권한부터 확인.

## 알려진 미확정 (실데이터로 보정 예정)

- 4회 구조 평탄 후보풀 vs 쌍유지 — 현재 평탄
- 캐릭터 폴백 threshold — 현재 DEFAULT_THRESHOLD
- 별칭매칭(bigram) 미구현 — "은발→grey_hair"가 top_k 밖이면 못 잡는 케이스 잔존
- 인원수 슬롯보호 효과 — 다인물 실데이터로 검증 필요
- 캐릭터/작품 분할입력(고급 모드) — 구현됨. 캐릭터 히트 threshold(CHAR_HIT_THRESHOLD, 현재 0.80)는 실데이터로 보정 필요

## 주의

- uvicorn 단일 프로세스(워커 1). 늘리면 인메모리 로그/캐시가 프로세스별로 갈림.
- reload 켜지 말 것(좀비 프로세스 + 멀티프로세스 로그 분산).
- CORS allow_origins 현재 "*". 같은 오리진 서빙이라 무방하나, 외부 노출 시 도메인으로 좁힐 것.
- 작가(cat1)·메타(cat5)는 인덱싱하도록 바뀌었으나(EXCLUDED_CATEGORIES 비움), **인덱스 재빌드를 해야 실제로 검색된다.** 재빌드 전에는 검색 풀/직접조회에서 작가·메타를 선택해도 0건. 5070Ti에서 빌드해 4060으로 복사(CPU 빌드 금지).

## 스트리밍 / 대기열 (nginx 설정 중요)

생성 요청은 4단계로 스트리밍된다(한국어키워드→영어키워드→최종태그→자연어).
동시 요청은 1개씩 직렬 처리하며, 대기 중인 사용자에게 "앞에 N명"을 실시간 표시한다.
(Redis 없이 프로세스 내 티켓 큐. 단일 워커 전제 — 워커 늘리면 큐가 갈라지니 1 유지.)

이게 동작하려면 리버스프록시가 응답을 버퍼링하면 안 된다. 두 가지로 보장:
1. 앱이 `X-Accel-Buffering: no` 헤더를 보냄 (nginx가 이 응답만 버퍼링 해제).
2. nginx 설정에서도 명시 (확실하게):

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
혹시 버퍼링되면 위 헤더가 도움이 된다. 스텝이 한꺼번에 뜨면 버퍼링을 의심.

