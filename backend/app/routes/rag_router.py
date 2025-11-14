from fastapi import APIRouter, Query, HTTPException
from pathlib import Path
from backend.app.model.step2_embedding import BGEEmbedding, ChromaVectorDB, search_qa
import logging
import os

# 라우터 정의
router = APIRouter(prefix="/rag", tags=["RAG Chatbot"])

# 프로젝트 루트 경로 설정 (Railway 환경 고려)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

# ChromaDB 경로 설정 (Railway 환경에서도 동작하도록)
CHROMA_DB_DIR = BACKEND_DIR / "chroma_db"
CHROMA_DB_PATH = str(CHROMA_DB_DIR)

# 전역 변수 (지연 로딩용)
embedder = None
vector_db = None
_initialization_error = None

def get_embedder():
    """임베딩 모델 지연 로딩 - 첫 요청 시에만 모델 다운로드 및 로드"""
    global embedder, _initialization_error
    
    if _initialization_error:
        raise _initialization_error
    
    if embedder is None:
        try:
            logging.info("🔄 임베딩 모델 초기화 중... (첫 요청 시 모델 다운로드로 시간이 걸릴 수 있습니다)")
            logging.info(f"   프로젝트 루트: {PROJECT_ROOT}")
            logging.info(f"   백엔드 디렉토리: {BACKEND_DIR}")
            embedder = BGEEmbedding(model_name="dragonkue/BGE-m3-ko")
            logging.info("✅ 임베딩 모델 로드 완료")
        except Exception as e:
            error_msg = f"임베딩 모델 로드 실패: {str(e)}"
            logging.error(f"❌ {error_msg}")
            logging.exception(e)
            _initialization_error = HTTPException(status_code=500, detail=error_msg)
            raise _initialization_error
    
    return embedder

def get_vector_db():
    """벡터DB 지연 로딩"""
    global vector_db, _initialization_error
    
    if _initialization_error:
        raise _initialization_error
    
    if vector_db is None:
        try:
            logging.info("🔄 벡터DB 초기화 중...")
            logging.info(f"   ChromaDB 경로: {CHROMA_DB_PATH}")
            logging.info(f"   경로 존재 여부: {os.path.exists(CHROMA_DB_PATH)}")
            
            # 디렉토리 생성 (없을 경우)
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            
            vector_db = ChromaVectorDB(
                collection_name="perso_qa_collection", 
                persist_dir=CHROMA_DB_PATH
            )
            
            doc_count = vector_db.count()
            logging.info(f"✅ 벡터DB 로드 완료 (문서 수: {doc_count})")
            
            if doc_count == 0:
                logging.warning("⚠️ ChromaDB에 문서가 없습니다. 임베딩 데이터를 먼저 생성해야 합니다.")
                
        except Exception as e:
            error_msg = f"벡터DB 초기화 실패: {str(e)}"
            logging.error(f"❌ {error_msg}")
            logging.exception(e)
            _initialization_error = HTTPException(status_code=500, detail=error_msg)
            raise _initialization_error
    
    return vector_db

@router.get("/query")
def query_rag(question: str = Query(..., description="사용자 질문")):
    """RAG 쿼리 엔드포인트"""
    try:
        logging.info(f"📥 쿼리 수신: {question}")
        
        # 첫 요청 시에만 모델 로드 (빌드 타임아웃 방지)
        embedder_instance = get_embedder()
        vector_db_instance = get_vector_db()
        
        # 검색 실행
        results = search_qa(question, embedder_instance, vector_db_instance, top_k=3)
        
        if not results:
            logging.warning(f"⚠️ 검색 결과 없음: {question}")
            return {
                "query": question,
                "best_answer": "관련된 정보를 찾을 수 없습니다.",
                "similarity": 0.0
            }
        
        best = results[0]
        logging.info(f"✅ 검색 완료 (유사도: {best.get('similarity', 0):.4f})")
        
        return {
            "query": question,
            "best_answer": best["answer"],
            "similarity": best["similarity"],
        }
        
    except HTTPException:
        # 이미 HTTPException이면 그대로 전달
        raise
    except Exception as e:
        error_msg = f"쿼리 처리 중 오류 발생: {str(e)}"
        logging.error(f"❌ {error_msg}")
        logging.exception(e)
        raise HTTPException(status_code=500, detail=error_msg)
