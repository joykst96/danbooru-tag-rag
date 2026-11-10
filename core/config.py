"""
설정 모듈

프로젝트 전역 상수를 정의한다.
경로, 임베딩 모델(embeddinggemma-2), 필터 정책, 인덱스, LLM 설정을 한곳에 모은다.
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
# google/embeddinggemma-2 (근거·실측: tag-embedding 벤치 README).
# embeddinggemma-2 설정 (모델 카드 + 벤치 실측으로 확인)
#   - 텍스트 인코더만 로드(vision/audio 제외 → 270M). float16 금지(NaN), fp32 로 고정.
#   - 쿼리·문서 모두 내장 프롬프트 SentenceSimilarity ("task: sentence similarity | query: ") — 카드의 대칭 태스크 방식.
#     문서 텍스트는 "{태그명} {한국어 정의} {한국어 별칭}" (core/parser.build_embed_text).
#     벤치 비교(2026-11-08): 단일 질의 MRR sim 0.739 / title 형식 0.590 / 검색용 비대칭 0.719 / 정의만 0.661,
#     서비스 시나리오(조각별 한국어+영어+어휘일치) 후보 회수율 sim 96% / 91% / 94% / 94%.
#   - max_seq 8192 를 명시해야 함(토크나이저 기본값은 1e30 센티널).
#   - 빌드 배치는 토큰 예산으로 나눔(기본 예산이면 peak VRAM 19GB).
GEMMA2_HF_ID = "google/embeddinggemma-2"
GEMMA2_LOCAL_NAME = "embeddinggemma-2"
GEMMA2_QUERY_PROMPT = "SentenceSimilarity"
GEMMA2_PASSAGE_PROMPT = "SentenceSimilarity"
GEMMA2_CONFIG_KWARGS = {"vision_config": None, "audio_config": None}
GEMMA2_MAX_SEQ_LENGTH = 8192
GEMMA2_TOKEN_BUDGET = 32768
# 실제 로드 경로: 로컬 폴더가 있으면 그걸, 없으면 HF ID로 폴백
def get_embedding_model_path() -> str:
    """로컬 모델 폴더가 존재하면 그 경로를, 없으면 HF ID 문자열을 반환."""
    local_path = MODELS_DIR / GEMMA2_LOCAL_NAME
    return str(local_path) if local_path.exists() else GEMMA2_HF_ID


# 디바이스: 'cuda' / 'cpu' / None(자동). 환경변수 EMBEDDING_DEVICE로 제어.
# 인덱스 빌드는 'cuda' 권장 — CPU 대비 수십 배 빠름.
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", None)

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
# 인덱스는 sim 하나.
#   sim = danbooru-tags.csv(mengod-hf/danbooru-tags-kr 2026-11-10판에서 core/tags_source.py 로 생성)의 태그를
#         "{태그명} {정의} {별칭}" 으로, 쿼리·문서 모두 SentenceSimilarity 프롬프트를 붙여 임베딩.
#         (벤치 이름 ns. 이전 커밋의 nt(title 형식)는 단일 질의·영어 질의에서 유의하게 나빠 교체했다.)
#   태그 118,133개(post_count >= 50, 11-04판은 116,325개). 경로는 모델별 폴더(data/embeddinggemma-2/lancedb_sim).
MAIN_VARIANT = "sim"


def get_lancedb_path(variant: str) -> Path:
    """variant 에 해당하는 LanceDB 디렉토리 경로."""
    return DATA_DIR / GEMMA2_LOCAL_NAME / f"lancedb_{variant}"


def get_table_name(variant: str) -> str:
    """variant에 해당하는 테이블명."""
    return f"danbooru_tags_{variant}"


# ---------------------------------------------------------------------------
# 검색 기본값
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 10
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

# ---------------------------------------------------------------------------
# 구조화 호출(structured 파이프라인의 분해·선택) 설정
# ---------------------------------------------------------------------------
#   temperature 0 + 고정 seed. thinking 은 선택 프롬프트에서 슬롯 컨텍스트(8192)를 다 써 40% 실패 → 항상 off.
#   프롬프트 캐시(LLM_CACHE_PROMPT, 기본 on): 켜면 호출당 3~5초 빠르고(시스템 프롬프트 재처리 없음),
#   두 번 실행 결과가 분해 8/10·선택 9/10 동일(채점 점수는 같음). off 면 10/10 동일하지만 느리다.
#   (실측: gemma4 26B A4B IQ4_XS, 벤치 2026-11-05)
LLM_SEED = int(os.environ.get("LLM_SEED", "1234"))
LLM_CACHE_PROMPT = os.environ.get("LLM_CACHE_PROMPT", "on").lower() not in ("off", "false", "0", "no", "")
