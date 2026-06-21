"""
설정 모듈

프로젝트 전역 상수를 정의한다.
경로, 모델, 필터 정책, 임베딩 a/b/c 인덱스 설정을 한곳에 모은다.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
# 프로젝트 루트: core/config.py → parents[1] = 레포 루트
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 루트의 .env 를 읽어 환경변수로 로드 (민감정보는 .env 에만 두고 git 에는 안 올림)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # python-dotenv 미설치 시 OS 환경변수에만 의존

# CSV 태그 파일 (루트에 위치)
TAGS_CSV_PATH = PROJECT_ROOT / "danbooru-tags.csv"

# 데이터 디렉토리 (LanceDB가 저장될 곳). Docker에서는 환경변수로 덮어씀.
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))

# 모델 로컬 저장 경로 (download_model.py 가 여기에 받아둔다)
MODELS_DIR = Path(os.environ.get("MODELS_DIR", PROJECT_ROOT / "models"))

# 생성 결과 로그 디렉토리 (사용자입력/최종태그/자연어 기록). Docker에서 rw 볼륨 마운트.
# 실제 파일 경로는 genlog 가 환경변수 GEN_LOG_PATH 로 직접 결정한다(여기 상수는 참고용).
LOG_DIR = Path(os.environ.get("LOG_DIR", PROJECT_ROOT / "logs"))

# ---------------------------------------------------------------------------
# 임베딩 모델 설정
# ---------------------------------------------------------------------------
# HuggingFace 모델 ID (다운로드 시 사용 / 로컬에 없을 때 폴백)
EMBEDDING_MODEL_HF_ID = "intfloat/multilingual-e5-large"

# 로컬에 받아둔 모델 폴더명
EMBEDDING_MODEL_LOCAL_NAME = "multilingual-e5-large"

# 실제 로드 경로: 로컬 폴더가 있으면 그걸, 없으면 HF ID로 폴백
def get_embedding_model_path() -> str:
    """로컬 모델 폴더가 존재하면 그 경로를, 없으면 HF ID 문자열을 반환."""
    local_path = MODELS_DIR / EMBEDDING_MODEL_LOCAL_NAME
    if local_path.exists():
        return str(local_path)
    return EMBEDDING_MODEL_HF_ID


EMBEDDING_DIM = 1024
MAX_SEQ_LENGTH = 512

# 디바이스: 'cuda' / 'cpu' / None(자동). 환경변수 EMBEDDING_DEVICE로 제어.
# 테스트 환경(eGPU RTX 4060)에서는 'cuda' 권장 — CPU 대비 임베딩 빌드가 수십 배 빠름.
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", None)

# E5 모델 비대칭 프리픽스
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# 임베딩 배치 크기
EMBED_BATCH_SIZE = 256

# ---------------------------------------------------------------------------
# 필터 정책 (스캔 데이터 분석으로 확정됨)
# ---------------------------------------------------------------------------
# category 의미: 0=일반, 1=아티스트, 3=작품/출처, 4=캐릭터, 5=메타
# 정책 변경(2분할 도입 이후): 카테고리는 빌드에서 드롭하지 않고 전부 인덱싱한다.
#   - 임베딩 벡터는 태그별 독립 계산 → 다른 카테고리가 섞여도 벡터값에 상호영향 없음.
#   - 프롬프트 파이프라인은 search 단계 where절(GENERAL_CATS{0}/CHARACTER_CATS{3,4})로
#     필요한 카테고리만 조회하므로, 작가(1)/메타(5)가 인덱스에 있어도 결과에 안 섞인다.
#   - cat1/cat5 는 직접조회(/api/direct_search)에서 사용자가 명시 선택할 때만 노출.
# (과거 cat1=저빈도 노이즈, cat5=내용무관 사유로 드롭했으나, where절 필터로 충분해 살림.)
EXCLUDED_CATEGORIES: set[int] = set()   # 빈 집합 = 전 카테고리 인덱싱

# 카테고리 한글 라벨 (검색 결과 분류/표시용)
CATEGORY_LABELS = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "meta",
}

# 빈도 필터: CSV가 이미 post_count 50 이상으로 잘려 있으므로 사실상 해제.
# 50 미만은 데이터에 존재하지 않음 (스캔으로 확인).
MIN_FREQUENCY = 50

# ---------------------------------------------------------------------------
# LanceDB 설정
# ---------------------------------------------------------------------------
# 임베딩 텍스트 구성이 다른 3개 인덱스를 만들어 비교한다.
#   variant A: 영어 태그명 + 한국어 정의 + 별칭  (전부 다 넣기)
#   variant B: 영어 태그명 + 한국어 별칭          (정의 제외)
#   variant C: 한국어 정의 + 별칭                  (영어 태그명 제외, 순수 한국어 공간)
# 비교 축:
#   A vs B → 정의문의 기여도
#   A vs C → 영어 태그명의 기여도
#   B vs C → 영어태그 vs 한국어정의 신호 강도
# 각 variant 는 별도 LanceDB 디렉토리 + 테이블로 저장되어 독립 비교 가능.
INDEX_VARIANTS = ("a", "b", "c")
DEFAULT_VARIANT = "a"


def get_lancedb_path(variant: str) -> Path:
    """variant('a'|'b')에 해당하는 LanceDB 디렉토리 경로."""
    return DATA_DIR / f"lancedb_{variant}"


def get_table_name(variant: str) -> str:
    """variant에 해당하는 테이블명."""
    return f"danbooru_tags_{variant}"


# ---------------------------------------------------------------------------
# 검색 기본값
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 10
MAX_TOP_K = 100
# 유사도 임계값 (distance_to_similarity 변환 후 0~1 스케일)
DEFAULT_THRESHOLD = 0.80

# ---------------------------------------------------------------------------
# LLM 설정 (로컬 llama.cpp 등 OpenAI 호환 엔드포인트)
# ---------------------------------------------------------------------------
LLM_API_URL = os.environ.get(
    "LLM_API_URL", "http://localhost:8080/v1/chat/completions"
)
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma4")

# thinking 활성 여부. 기본 off (변별 작업이라 사고과정 불필요, 속도 우선).
LLM_THINKING_ENABLED = os.environ.get("LLM_THINKING", "off").lower() not in (
    "off", "false", "0", "no", ""
)
