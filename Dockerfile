# danbooru-tags-rag 배포 이미지 (CPU 추론)
# 모델/LanceDB/CSV/.env 는 이미지에 굽지 않고 볼륨 마운트한다.
FROM python:3.12-slim

# uv 설치 (빠른 의존성 설치)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 의존성 메타만 먼저 복사해 레이어 캐시 활용
COPY pyproject.toml ./
COPY uv.lock* ./

# CPU torch 인덱스 고정 + 의존성 설치
# (pyproject 에 torch CPU 인덱스를 명시해두는 것이 가장 깔끔하지만,
#  안전하게 여기서도 CPU 휠을 강제한다)
ENV UV_LINK_MODE=copy
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch \
 && uv pip install --system -r pyproject.toml

# 애플리케이션 코드
COPY core/ ./core/
COPY index.html ./

# 런타임 환경: CPU 임베딩 강제, 데이터 경로는 마운트 위치로
ENV EMBEDDING_DEVICE=cpu \
    DATA_DIR=/app/data \
    MODELS_DIR=/app/models \
    PYTHONUNBUFFERED=1

EXPOSE 3333

# uvicorn 단일 프로세스(reload 끔). 워커 늘리면 인메모리 로그/캐시 분산되니 1로.
CMD ["python", "-m", "core.api"]
