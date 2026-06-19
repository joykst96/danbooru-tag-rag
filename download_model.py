"""
모델 다운로드 스크립트 (최초 1회 실행)

HuggingFace에서 임베딩 모델을 받아 로컬 ./models/ 폴더에 통째로 저장한다.
이후 config.get_embedding_model_path() 가 이 로컬 경로를 자동으로 사용하므로,
서버/빌더를 아무리 자주 재시작해도 모델을 다시 받지 않는다.

사용법:
    python download_model.py

Docker 환경에서는 ./models 를 볼륨 마운트하여 컨테이너가 공유한다.
"""

import sys
from pathlib import Path

# config 의 상수를 재사용 (단일 출처 유지)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from danbooru_rag.config import (
    EMBEDDING_MODEL_HF_ID,
    EMBEDDING_MODEL_LOCAL_NAME,
    MODELS_DIR,
)


def main():
    target_dir = MODELS_DIR / EMBEDDING_MODEL_LOCAL_NAME

    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"이미 받아져 있음: {target_dir}")
        print("다시 받으려면 해당 폴더를 지우고 재실행하세요.")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"모델 다운로드 시작: {EMBEDDING_MODEL_HF_ID}")
    print(f"저장 위치: {target_dir}")

    # sentence-transformers 로 받아서 그대로 로컬에 저장.
    # huggingface_hub.snapshot_download 도 가능하지만,
    # SentenceTransformer.save() 를 쓰면 ST가 바로 로드 가능한 형태로 정리되어 깔끔하다.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_HF_ID)
    model.save(str(target_dir))

    print("=" * 50)
    print(f"다운로드 완료: {target_dir}")
    print("이제 config 가 이 로컬 경로를 자동으로 사용합니다.")
    print("=" * 50)


if __name__ == "__main__":
    main()
