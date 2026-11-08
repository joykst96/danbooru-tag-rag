"""
Danbooru Tag RAG

한국어 입력 → 구조화 분해 → 조각별 벡터·어휘 검색 → 후보 번호 선택 → Danbooru 태그 + 자연어 프롬프트.
로컬 임베딩(google/embeddinggemma-2) + 로컬 LLM(llama.cpp, OpenAI 호환) 기반.
"""

__version__ = "0.1.0"
