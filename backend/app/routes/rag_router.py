from fastapi import APIRouter, Query
from pathlib import Path
from backend.app.model.step2_embedding import BGEEmbedding, ChromaVectorDB, search_qa
import logging

# 라우터 정의
router = APIRouter(prefix="/rag", tags=["RAG Chatbot"])

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

# 전역 변수 (지연 로딩용)
embedder = None
vector_db = None

def get_embedder():
    """임베딩 모델 지연 로딩 - 첫 요청 시에만 모델 다운로드 및 로드"""
    global embedder
    if embedder is None:
        logging.info("🔄 임베딩 모델 초기화 중... (첫 요청 시 모델 다운로드로 시간이 걸릴 수 있습니다)")
        embedder = BGEEmbedding(model_name="dragonkue/BGE-m3-ko")
        logging.info("✅ 임베딩 모델 로드 완료")
    return embedder

def get_vector_db():
    """벡터DB 지연 로딩"""
    global vector_db
    if vector_db is None:
        logging.info("🔄 벡터DB 초기화 중...")
        vector_db = ChromaVectorDB(
            collection_name="perso_qa_collection", 
            persist_dir=str(BACKEND_DIR / "chroma_db")
        )
        logging.info("✅ 벡터DB 로드 완료")
    return vector_db

@router.get("/query")
def query_rag(question: str = Query(..., description="사용자 질문")):
    # 첫 요청 시에만 모델 로드 (빌드 타임아웃 방지)
    embedder_instance = get_embedder()
    vector_db_instance = get_vector_db()
    
    results = search_qa(question, embedder_instance, vector_db_instance, top_k=3)
    if not results:
        return {"answer": "관련된 정보를 찾을 수 없습니다."}
    best = results[0]
    return {
        "query": question,
        "best_answer": best["answer"],
        "similarity": best["similarity"],
    }
