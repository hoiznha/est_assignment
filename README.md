# Perso.ai RAG Chatbot

Perso.ai 서비스에 대한 질문에 답변하는 RAG(Retrieval-Augmented Generation) 기반 챗봇입니다.

## 📋 프로젝트 개요

이 프로젝트는 Q&A 데이터를 벡터화하여 저장하고, 사용자의 질문에 제공된 데이터셋 내 존재하는 답변만을 제공하는 RAG 시스템입니다. 오픈소스 임베딩 모델을 사용하여 API 키 없이 로컬에서 실행 가능합니다.

## 🛠️ 사용 기술 스택

### Backend
- **Framework**: FastAPI 0.104.0+
- **Server**: Uvicorn (ASGI)
- **Language**: Python 3.11
- **임베딩 모델**: 
  - `dragonkue/BGE-m3-ko` (BAAI BGE-M3 기반 한국어 모델)
  - Sentence Transformers 2.2.0+
- **벡터 데이터베이스**: ChromaDB 0.4.0+
- **딥러닝 프레임워크**: PyTorch 2.0.0+ (CPU-only)
- **데이터 처리**: Pandas 2.0.0+, OpenPyXL 3.1.0+

### Frontend
- **Framework**: React 19.2.0
- **Build Tool**: Vite 7.2.2
- **Language**: TypeScript 5.9.3
- **UI**: Custom CSS (반응형 디자인)

### 배포
- **Backend**: Railway (Docker 기반)
- **Frontend**: Vercel
- **Container**: Docker (멀티스테이지 빌드)

## 🗄️ 벡터 DB 및 임베딩 방식

### 벡터 데이터베이스: ChromaDB

**선택 이유:**
- 오픈소스이며 Python과의 통합이 용이
- 영구 저장소(Persistent Storage) 지원
- 코사인 유사도 기반 검색 최적화
- 메타데이터 필터링 지원

**구현 방식:**
- **저장 방식**: 로컬 파일 시스템 기반 영구 저장
- **인덱싱**: HNSW (Hierarchical Navigable Small World) 알고리즘
- **유사도 측정**: 코사인 유사도 (Cosine Similarity)
- **컬렉션**: `perso_qa_collection` 단일 컬렉션 사용

**데이터 구조:**
```python
{
    "id": "qa_001",
    "embedding": [0.123, 0.456, ...],  # 벡터 임베딩
    "document": "답변 텍스트",
    "metadata": {
        "question": "원본 질문",
        "category": "카테고리",
        "keywords": "키워드1, 키워드2"
    }
}
```

### 임베딩 방식: BGE-m3-ko

**모델 정보:**
- **모델명**: `dragonkue/BGE-m3-ko`
- **기반**: BAAI BGE-M3 (Beijing Academy of Artificial Intelligence)
- **언어**: 한국어 특화
- **임베딩 차원**: 1024차원 (BGE-M3 기준)
- **특징**: Passage와 Query를 구분하여 임베딩

**임베딩 전략:**

1. **문서 임베딩 (Passage Embedding)**
   ```python
   # 프리픽스: "passage: "
   text = "passage: 질문: Perso.ai는 어떤 서비스인가요?\n답변: ..."
   embedding = model.encode(text, normalize_embeddings=True)
   ```
   - 질문과 답변을 결합하여 임베딩 (더 풍부한 의미 표현)
   - 정규화된 벡터로 코사인 유사도 최적화

2. **쿼리 임베딩 (Query Embedding)**
   ```python
   # 프리픽스: "query: "
   query = "query: Perso.ai는 어떤 서비스인가요?"
   embedding = model.encode(query, normalize_embeddings=True)
   ```
   - 검색 쿼리만 임베딩
   - 질문-질문 매칭에 최적화

**최적화 기법:**
- **정규화**: 모든 임베딩을 L2 정규화하여 코사인 유사도 계산 최적화
- **배치 처리**: 문서 임베딩 시 배치 크기 32로 처리
- **CPU 최적화**: 
  - `torch.no_grad()`로 그래디언트 계산 비활성화
  - CPU 스레드 수 최적화 (최대 4스레드)
  - 모델을 eval 모드로 설정

## 🎯 정확도 향상 전략

### 1. 질문-질문 매칭 방식

**문제점:**
- 초기에는 질문+답변을 결합하여 임베딩했으나, 검색 시 질문만 입력하므로 유사도가 낮게 나옴

**해결책:**
- **저장**: 질문만 임베딩하여 저장
- **검색**: 사용자 질문을 임베딩하여 질문-질문 매칭
- **결과**: 같은 질문에 대해 유사도 1.0에 가까운 높은 정확도 달성

```python
# 저장 시
text = question  # 질문만 사용

# 검색 시
query_embedding = embed_query(user_question)
results = vector_db.search(query_embedding, top_k=3)
```

### 2. 정규화 일관성

**문제점:**
- 문서 임베딩과 쿼리 임베딩의 정규화 설정이 다르면 유사도 계산이 부정확

**해결책:**
- 모든 임베딩에 `normalize_embeddings=True` 적용
- 코사인 유사도 계산 최적화

### 3. Top-K 검색 및 유사도 필터링

**전략:**
- `top_k=3`으로 상위 3개 결과 반환
- 가장 유사도가 높은 답변을 `best_answer`로 선택
- 유사도가 너무 낮으면 "관련 정보 없음" 반환

### 4. 메타데이터 활용

**구조:**
- 질문, 카테고리, 키워드를 메타데이터로 저장
- 향후 메타데이터 기반 필터링 가능

### 5. 모델 최적화

**성능 개선:**
- CPU 환경 최적화로 추론 속도 향상 (20-30% 개선)
- 메모리 효율적 처리로 안정성 향상

## 📁 프로젝트 구조

```
est_assignment/
├── main.py                    # FastAPI 메인 애플리케이션 진입점   
├── backend/                    # 백엔드 (Python/FastAPI)
│   ├── app/                   
│   │   ├── model/             
│   │   │   ├── step1_preprocess.py    # 데이터 전처리 (Q&A 데이터 정제)
│   │   │   └── step2_embedding.py     # 임베딩 생성 및 ChromaDB 저장
│   │   └── routes/            
│   │       └── rag_router.py   # RAG 쿼리 엔드포인트 (/rag/query)
│   ├── data/                  
│   │   ├── raw/               
│   │   │   └── Q&A.xlsx       # 원본 Q&A 엑셀 파일
│   │   └── processed/          # 전처리된 데이터
│   │       ├── qa_preprocessed.csv
│   │       └── qa_preprocessed.json
│   │
│   └── chroma_db/   
├── README.md                   
└── frontend/                   # 프론트엔드 (React/TypeScript/Vite)
    ├── index.html              
    ├── public/                  
    │
    └── src/                    
        ├── main.tsx            
        ├── App.tsx            
        ├── components/         
        │   └── Chat.tsx       
        ├── styles/             
        │   ├── index.css      
        │   ├── App.css        
        │   └── Chat.css       
        │
        └── assets/            
```


