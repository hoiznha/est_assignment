from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routes.rag_router import router as rag_router
import uvicorn
import os
import threading
import logging

app = FastAPI(title="RAG Chatbot API")

# CORS 설정 - 명시적으로 허용할 origin 지정
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://est-assignment.vercel.app",
    os.getenv("FRONTEND_URL", ""),  # 환경 변수로 추가 도메인 설정 가능
]

allowed_origins = [origin for origin in allowed_origins if origin]

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check 엔드포인트 
@app.get("/")
@app.get("/health")
def health_check():
    """Health check - 모델 로딩과 무관하게 즉시 응답"""
    return {
        "status": "ok", 
        "message": "API is running",
        "service": "RAG Chatbot API"
    }

# 라우터 등록
app.include_router(rag_router)

# 서버 시작 시 모델 사전 로딩 (백그라운드)
@app.on_event("startup")
def startup_event():
    """서버 시작 시 백그라운드에서 모델 사전 로딩"""
    def preload_models():
        try:
            logging.info("🚀 서버 시작: 백그라운드에서 모델 사전 로딩 시작...")
            from backend.app.routes.rag_router import get_embedder, get_vector_db
            
            # 모델과 벡터DB 사전 로딩
            embedder = get_embedder()
            vector_db = get_vector_db()
            
            doc_count = vector_db.count()
            logging.info(f"✅ 모델 사전 로딩 완료! (ChromaDB 문서 수: {doc_count})")
            logging.info("   이제 API 요청이 즉시 처리됩니다.")
            
        except Exception as e:
            logging.warning(f"⚠️ 모델 사전 로딩 실패 (첫 요청 시 로드됨): {str(e)}")
            logging.exception(e)
    
    # 백그라운드 스레드에서 실행 (서버 시작을 블로킹하지 않음)
    thread = threading.Thread(target=preload_models, daemon=True)
    thread.start()
    logging.info("📡 FastAPI 서버 시작됨 (모델은 백그라운드에서 로딩 중...)")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
