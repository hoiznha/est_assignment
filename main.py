from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routes.rag_router import router as rag_router
import uvicorn
import os

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
    return {"status": "ok", "message": "API is running"}

# 라우터 등록
app.include_router(rag_router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
