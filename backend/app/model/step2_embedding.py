#step2_embedding.py

"""
BGE-m3-ko 모델을 사용한 임베딩 생성 및 ChromaDB 저장
- 모델: dragonkue/BGE-m3-ko (오픈소스)
- 로컬 실행 가능 (API 키 불필요)
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import torch  # 모델 최적화를 위해 필요

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# =============================================================================
# BGE-m3-ko 임베딩 클래스
# =============================================================================

class BGEEmbedding:
    """
    BGE-m3-ko 오픈소스 임베딩 모델 핸들러
    HuggingFace Transformers 기반
    """

    def __init__(self, model_name: str = "dragonkue/BGE-m3-ko", device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "❌ sentence-transformers를 설치하세요:\n"
                "pip install sentence-transformers"
            )
        
        import torch
        
        self.model_name = model_name
        
        # 디바이스 설정 (CUDA, MPS, CPU)
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # CPU 환경 최적화 설정
        if self.device == "cpu":
            # CPU 스레드 수 최적화 (Railway 환경 고려)
            num_threads = min(4, torch.get_num_threads())  # 최대 4스레드
            torch.set_num_threads(num_threads)
            logging.info(f"   CPU 스레드 수: {num_threads}")
        
        logging.info(f"🔄 BGE-m3-ko 모델 로딩 중... (디바이스: {self.device})")
        logging.info("   ⏳ 첫 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다...")
        
        # 모델 로드 최적화 옵션
        model_kwargs = {}
        if self.device == "cuda":
            # CUDA 환경: FP16 사용 (메모리 절약 및 속도 향상)
            try:
                model_kwargs = {'torch_dtype': torch.float16}
                logging.info("   최적화: FP16 사용 (CUDA)")
            except:
                pass
        elif self.device == "cpu":
            # CPU 환경: 모델을 eval 모드로 설정하고 최적화
            model_kwargs = {}
            logging.info("   최적화: CPU 추론 최적화")
        
        # 모델 로드
        self.model = SentenceTransformer(
            model_name, 
            device=self.device,
            model_kwargs=model_kwargs if model_kwargs else None
        )
        
        # 모델을 평가 모드로 설정 (드롭아웃 등 비활성화)
        self.model.eval()
        
        # CPU 환경에서 추가 최적화
        if self.device == "cpu":
            # torch.jit.script로 최적화 시도 (선택적)
            try:
                # 모델의 일부를 최적화할 수 있지만, sentence-transformers는 이미 최적화되어 있음
                pass
            except:
                pass
        
        logging.info(f"✅ BGE-m3-ko 모델 로드 완료!")
        logging.info(f"   모델: {model_name}")
        logging.info(f"   디바이스: {self.device}")
        logging.info(f"   최적화: 활성화됨")

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        문서 임베딩 (배치 처리)
        
        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기
            
        Returns:
            임베딩 벡터 리스트
        """
        if not texts:
            logging.warning("⚠️ 임베딩할 텍스트가 없습니다.")
            return []
        
        logging.info(f"📊 {len(texts)}개 문서 임베딩 생성 중...")
        
        # BGE 모델은 retrieval을 위해 "passage: " 프리픽스 사용
        passages = [f"passage: {text}" for text in texts]
        
        # 배치 임베딩 (최적화 옵션 적용)
        # CPU 환경에서 메모리 효율적 처리
        with torch.no_grad():  # 그래디언트 계산 비활성화로 메모리 절약 및 속도 향상
            embeddings = self.model.encode(
                passages,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # 코사인 유사도 최적화
            )
        
        # numpy array를 list로 변환
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        logging.info(f"✅ {len(embeddings_list)}개 임베딩 생성 완료")
        logging.info(f"📐 벡터 차원: {len(embeddings_list[0]) if embeddings_list else 0}")
        
        return embeddings_list

    def embed_query(self, query: str) -> Optional[List[float]]:
        """
        검색 쿼리 임베딩
        
        Args:
            query: 검색 쿼리
            
        Returns:
            임베딩 벡터
        """
        if not query or not query.strip():
            logging.warning("⚠️ 빈 쿼리입니다.")
            return None
        
        # BGE 모델은 검색 쿼리에 "query: " 프리픽스 사용
        query_with_prefix = f"query: {query}"
        
        # 단일 임베딩 (최적화 옵션 적용)
        # CPU 환경에서 메모리 효율적 처리 및 속도 향상
        with torch.no_grad():  # 그래디언트 계산 비활성화로 메모리 절약 및 속도 향상
            embedding = self.model.encode(
                query_with_prefix,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=1  # 단일 쿼리 최적화
            )
        
        return embedding.tolist()

    def get_embedding_dimension(self) -> int:
        """임베딩 차원 반환"""
        return self.model.get_sentence_embedding_dimension()


# =============================================================================
# ChromaDB 핸들러
# =============================================================================

class ChromaVectorDB:
    """ChromaDB 영구 벡터 저장소"""

    def __init__(
        self, 
        collection_name: str = "perso_qa_collection", 
        persist_dir: str = "/chroma_db",
        recreate: bool = False
    ):
        try:
            import chromadb
        except ImportError:
            raise ImportError("❌ chromadb를 설치하세요: pip install chromadb")

        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        os.makedirs(persist_dir, exist_ok=True)
        
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(path=persist_dir)

        # 컬렉션 재생성
        if recreate:
            try:
                self.client.delete_collection(collection_name)
                logging.info(f"🧹 기존 컬렉션 '{collection_name}' 삭제 완료")
            except Exception:
                pass

        # 컬렉션 생성 또는 로드
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 코사인 유사도
        )
        
        logging.info(f"✅ ChromaDB 초기화 완료")
        logging.info(f"   컬렉션: {collection_name}")
        logging.info(f"   저장 위치: {persist_dir}")
        logging.info(f"   기존 문서 수: {self.collection.count()}")

    def add_documents(
        self, 
        ids: List[str], 
        embeddings: List[List[float]], 
        documents: List[str], 
        metadatas: List[Dict]
    ):
        """문서 추가"""
        
        # 입력 검증
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError(
                f"❌ 입력 길이 불일치: ids={len(ids)}, embeddings={len(embeddings)}, "
                f"documents={len(documents)}, metadatas={len(metadatas)}"
            )
        
        if not embeddings:
            raise ValueError("❌ 임베딩이 비어있습니다.")
        
        logging.info(f"💾 {len(ids)}개 문서를 ChromaDB에 저장 중...")
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logging.info(f"✅ {len(ids)}개 문서 저장 완료")
            
        except Exception as e:
            logging.error(f"❌ 문서 저장 실패: {e}")
            raise

    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 3,
        filter_metadata: Optional[Dict] = None
    ) -> Dict:
        """유사도 검색"""
        
        if not query_embedding:
            logging.error("❌ 유효하지 않은 쿼리 임베딩")
            return {
                'ids': [[]],
                'distances': [[]],
                'documents': [[]],
                'metadatas': [[]],
                'similarities': [[]]
            }
        
        # 검색 실행
        results = self.collection.query(
            query_embeddings=[query_embedding], 
            n_results=top_k,
            where=filter_metadata  # 메타데이터 필터링 (선택적)
        )
        
        # 유사도 계산 (거리 -> 유사도 변환)
        # ChromaDB의 코사인 거리: distance = 1 - cosine_similarity
        # 따라서 similarity = 1 - distance
        similarities = [[1 - d for d in results["distances"][0]]]
        results["similarities"] = similarities
        
        return results

    def count(self) -> int:
        """저장된 문서 수"""
        return self.collection.count()


# =============================================================================
# 메인 파이프라인
# =============================================================================

def create_embeddings_and_store(
    qa_json_path: str,
    chroma_dir: str = "./chroma_db",
    collection_name: str = "perso_qa_collection",
    model_name: str = "dragonkue/BGE-m3-ko",
    recreate: bool = True
) -> Tuple[BGEEmbedding, ChromaVectorDB]:
    """
    BGE-m3-ko 임베딩 생성 및 ChromaDB 저장 파이프라인
    
    Args:
        qa_json_path: Q&A JSON 파일 경로
        chroma_dir: ChromaDB 저장 디렉토리
        collection_name: 컬렉션 이름
        model_name: 임베딩 모델 이름
        recreate: 기존 컬렉션 재생성 여부
        
    Returns:
        (BGEEmbedding, ChromaVectorDB) 튜플
    """
    
    logging.info("="*60)
    logging.info("🚀 BGE-m3-ko 임베딩 생성 파이프라인 시작")
    logging.info("="*60)
    
    # 1. BGE 임베딩 모델 로드
    embedder = BGEEmbedding(model_name)
    
    # 2. ChromaDB 초기화
    vector_db = ChromaVectorDB(collection_name, chroma_dir, recreate=recreate)

    # 3. Q&A 데이터 로드
    logging.info(f"\n📂 Q&A 데이터 로드 중: {qa_json_path}")
    
    if not os.path.exists(qa_json_path):
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {qa_json_path}")
    
    with open(qa_json_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    
    logging.info(f"✅ {len(qa_data)}개 Q&A 데이터 로드 완료")

    # 4. 데이터 준비
    texts, ids, docs, metas = [], [], [], []
    
    for qa in qa_data:
        question = qa.get('question', '')
        answer = qa.get('answer', '')
        
        # 질문 + 답변 결합 (더 나은 검색 성능)
        text = f"질문: {question}\n답변: {answer}"
        
        texts.append(text)
        ids.append(qa.get('id', f"qa_{len(ids):03d}"))
        docs.append(answer)  # 실제 답변만 저장
        
        metadata = qa.get('metadata', {})
        metas.append({
            "question": str(question),
            "category": str(qa.get("category", "")),
            "keywords": ", ".join(metadata.get("keywords", [])),
            "answer_length": int(metadata.get("answer_length", 0))
        })
    
    logging.info(f"📝 {len(texts)}개 텍스트 준비 완료")

    # 5. 임베딩 생성
    logging.info("\n🔄 BGE-m3-ko로 임베딩 생성 중...")
    embeddings = embedder.embed_documents(texts, batch_size=32)
    
    if len(embeddings) != len(texts):
        logging.warning(
            f"⚠️ 임베딩 개수 불일치: 텍스트={len(texts)}, 임베딩={len(embeddings)}"
        )
    
    # 6. ChromaDB에 저장
    logging.info("\n💾 ChromaDB에 저장 중...")
    vector_db.add_documents(ids, embeddings, docs, metas)
    
    # 7. 최종 통계
    logging.info("\n" + "="*60)
    logging.info("📊 최종 통계")
    logging.info("="*60)
    logging.info(f"총 문서 수: {vector_db.count()}")
    logging.info(f"벡터 차원: {embedder.get_embedding_dimension()}")
    logging.info(f"모델: {model_name}")
    logging.info(f"저장 위치: {chroma_dir}")
    logging.info(f"컬렉션: {collection_name}")
    logging.info("="*60)

    return embedder, vector_db


# =============================================================================
# 검색 함수
# =============================================================================

def search_qa(
    query: str, 
    embedder: BGEEmbedding, 
    vector_db: ChromaVectorDB, 
    top_k: int = 3
) -> List[Dict]:
    """
    Q&A 검색
    
    Args:
        query: 사용자 질문
        embedder: BGEEmbedding 인스턴스
        vector_db: ChromaVectorDB 인스턴스
        top_k: 반환할 결과 개수
        
    Returns:
        검색 결과 리스트
    """
    # 쿼리 임베딩
    query_emb = embedder.embed_query(query)
    
    if not query_emb:
        logging.error("❌ 쿼리 임베딩 생성 실패")
        return []
    
    # 검색 실행 
    results = vector_db.search(query_emb, top_k)
    
    # 결과 포맷팅
    formatted = []
    for doc, meta, sim, doc_id in zip(
        results["documents"][0], 
        results["metadatas"][0], 
        results["similarities"][0],
        results["ids"][0]
    ):
        formatted.append({
            "id": doc_id,
            "question": meta.get("question"),
            "answer": doc,
            "similarity": sim,
            "category": meta.get("category"),
            "keywords": meta.get("keywords", "")
        })
    
    return formatted


# =============================================================================
# main()
# =============================================================================

def get_project_root() -> Path:
    """프로젝트 루트 경로 반환"""
    # 이 파일이 backend/ 폴더에 있으므로, parent.parent가 프로젝트 루트
    return Path(__file__).resolve().parent.parent.parent.parent


def main():
    """메인 실행 함수"""
    
    # 프로젝트 루트 경로 설정
    project_root = get_project_root()
    backend_dir = project_root / "backend"
    data_dir = backend_dir / "data" / "processed"
    chroma_dir_path = backend_dir / "chroma_db"
    
    # 설정
    qa_json_path = str(data_dir / "qa_preprocessed.json")
    chroma_dir = str(chroma_dir_path)
    collection_name = "perso_qa_collection"
    model_name = "dragonkue/BGE-m3-ko"
    
    # 파일 존재 확인
    if not os.path.exists(qa_json_path):
        logging.error(f"❌ Q&A 파일이 없습니다: {qa_json_path}")
        logging.error(f"   프로젝트 루트: {project_root}")
        logging.error(f"   예상 위치: {qa_json_path}")
        logging.info("💡 먼저 step1_preprocess.py를 실행하여 qa_preprocessed.json 파일을 생성하세요.")
        return
    
    # 임베딩 생성 및 저장
    try:
        embedder, vector_db = create_embeddings_and_store(
            qa_json_path=qa_json_path,
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            model_name=model_name,
            recreate=True
        )
        
        # 테스트 검색
        logging.info("\n" + "="*60)
        logging.info("🔍 테스트 검색")
        logging.info("="*60)
        
        test_queries = [
            "Perso.ai는 어떤 서비스인가요?"
            # "회원가입이 필요한가요?",
            # "어떤 언어를 지원하나요?"
            # "유튜버들이 이 서비스를 쓰는 이유가 무엇이며, 어떤 기술을 활용하나요?"
        ]
        
        for query in test_queries:
            logging.info(f"\n질문: '{query}'")
            logging.info("-"*60)
            
            results = search_qa(query, embedder, vector_db, top_k=3)
            
            for i, r in enumerate(results, 1):
                print(f"\n[{i}] 유사도: {r['similarity']:.4f} | ID: {r['id']}")
                print(f"    질문: {r['question']}")
                print(f"    답변: {r['answer'][:80]}...")
                print(f"    카테고리: {r['category']}")
        
        logging.info("\n" + "="*60)
        logging.info("✅ 모든 작업 완료!")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"❌ 파이프라인 실행 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()