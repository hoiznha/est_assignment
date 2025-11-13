from fastapi import APIRouter, Query
from pathlib import Path
from backend.app.model.step2_embedding import BGEEmbedding, ChromaVectorDB, search_qa

# 라우터 정의
router = APIRouter(prefix="/rag", tags=["RAG Chatbot"])

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

# 임베딩 및 벡터DB 로드
embedder = BGEEmbedding(model_name="dragonkue/BGE-m3-ko")
vector_db = ChromaVectorDB(
    collection_name="perso_qa_collection", 
    persist_dir=str(BACKEND_DIR / "chroma_db")
)

@router.get("/query")
def query_rag(question: str = Query(..., description="사용자 질문")):
    results = search_qa(question, embedder, vector_db, top_k=3)
    if not results:
        return {"answer": "관련된 정보를 찾을 수 없습니다."}
    best = results[0]
    return {
        "query": question,
        "best_answer": best["answer"],
        "similarity": best["similarity"],
    }
