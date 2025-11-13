#step1_preprocesse.py
"""
Perso.ai Q&A 데이터 전처리 스크립트
- 엑셀 데이터 읽기
- Q&A 추출 및 정제
- 메타데이터 추가
- JSON/CSV 형식으로 저장
"""

import pandas as pd
import json
import re
from datetime import datetime
from typing import List, Dict
import unicodedata
from pathlib import Path


class QAPreprocessor:
    """Q&A 데이터 전처리 클래스"""
    
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.qa_pairs = []
        
    def extract_qa_pairs(self) -> List[Dict]:
        """엑셀에서 Q&A 쌍 추출"""
        df = pd.read_excel(self.excel_path, sheet_name='샘플 데이터', header=None)
        content_col = df.iloc[:, -1]
        
        qa_pairs = []
        current_q = None
        
        for idx, content in enumerate(content_col):
            if pd.isna(content):
                continue
            
            content_str = str(content).strip()
            
            if content_str.startswith('Q.'):
                current_q = content_str[2:].strip()
            elif content_str.startswith('A.') and current_q:
                current_a = content_str[2:].strip()
                qa_pairs.append({
                    'question': current_q,
                    'answer': current_a
                })
                current_q = None
        
        return qa_pairs
    
    def normalize_text(self, text: str) -> str:
        """텍스트 정규화"""
        # 유니코드 정규화
        text = unicodedata.normalize('NFKC', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    def extract_keywords(self, text: str) -> List[str]:
        """간단한 키워드 추출 (명사 기반)"""
        # 실제로는 KoNLPy 등을 사용하는 것이 좋지만, 의존성을 줄이기 위해 간단한 패턴 사용
        keywords = []
        
        # 고유명사 추출 패턴
        patterns = [
            r'Perso\.ai',
            r'이스트소프트',
            r'ESTsoft',
            r'AI',
            r'더빙',
            r'음성',
            r'영상',
            r'립싱크',
            r'다국어'
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                keywords.append(pattern.replace(r'\.', '.'))
        
        return list(set(keywords))  # 중복 제거
    
    def categorize_qa(self, question: str, answer: str) -> str:
        """Q&A 카테고리 자동 분류"""
        text = question + " " + answer
        
        if any(word in text for word in ['서비스', '플랫폼', '기능', '주요']):
            return '서비스 소개'
        elif any(word in text for word in ['사용자', '고객', '유튜버', '기업']):
            return '사용자 정보'
        elif any(word in text for word in ['언어', '요금', '기술', '파트너']):
            return '기술 상세'
        elif any(word in text for word in ['이스트소프트', '회사', '개발', '설립']):
            return '회사 정보'
        elif any(word in text for word in ['가입', '사용', '문의', '편집']):
            return '사용 가이드'
        else:
            return '기타'
    
    def add_metadata(self, qa_pairs: List[Dict]) -> List[Dict]:
        """메타데이터 추가"""
        enriched_data = []
        
        for idx, qa in enumerate(qa_pairs, 1):
            question = self.normalize_text(qa['question'])
            answer = self.normalize_text(qa['answer'])
            
            enriched_qa = {
                'id': f'qa_{idx:03d}',
                'question': question,
                'answer': answer,
                'category': self.categorize_qa(question, answer),
                'metadata': {
                    'answer_length': len(answer),
                    'keywords': self.extract_keywords(question + " " + answer),
                    'created_at': datetime.now().isoformat()
                }
            }
            
            enriched_data.append(enriched_qa)
        
        return enriched_data
    
    def generate_question_variations(self, question: str) -> List[str]:
        """질문 변형 생성 (규칙 기반)"""
        variations = [question]  # 원본 포함
        
        # 의문사 변형
        variations_map = {
            '어떤': ['무슨', '어느'],
            '무엇인가요': ['뭐예요', '뭔가요'],
            '몇 개인가요': ['몇개인가요', '몇 개예요'],
            '어떻게': ['어찌', '어떻게'],
        }
        
        for original, replacements in variations_map.items():
            if original in question:
                for replacement in replacements:
                    variations.append(question.replace(original, replacement))
        
        return list(set(variations))  # 중복 제거
    
    def process(self, augment_questions: bool = False) -> List[Dict]:
        """전체 전처리 프로세스 실행"""
        print("📖 Q&A 데이터 추출 중...")
        self.qa_pairs = self.extract_qa_pairs()
        print(f"✅ {len(self.qa_pairs)}개의 Q&A 쌍 추출 완료")
        
        print("\n🔧 메타데이터 추가 중...")
        self.qa_pairs = self.add_metadata(self.qa_pairs)
        print("✅ 메타데이터 추가 완료")
        
        if augment_questions:
            print("\n🔄 질문 변형 생성 중...")
            augmented_data = []
            for qa in self.qa_pairs:
                variations = self.generate_question_variations(qa['question'])
                for var in variations:
                    augmented_qa = qa.copy()
                    augmented_qa['question'] = var
                    augmented_qa['is_variation'] = (var != qa['question'])
                    augmented_data.append(augmented_qa)
            self.qa_pairs = augmented_data
            print(f"✅ 총 {len(self.qa_pairs)}개로 확장 완료")
        
        return self.qa_pairs
    
    def save_to_json(self, output_path: str):
        """JSON 형식으로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, ensure_ascii=False, indent=2)
        print(f"\n💾 JSON 저장 완료: {output_path}")
    
    def save_to_csv(self, output_path: str):
        """CSV 형식으로 저장"""
        # 평면 구조로 변환
        flat_data = []
        for qa in self.qa_pairs:
            flat_data.append({
                'id': qa['id'],
                'question': qa['question'],
                'answer': qa['answer'],
                'category': qa['category'],
                'answer_length': qa['metadata']['answer_length'],
                'keywords': ', '.join(qa['metadata']['keywords'])
            })
        
        df = pd.DataFrame(flat_data)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"💾 CSV 저장 완료: {output_path}")
    
    def print_summary(self):
        """전처리 결과 요약 출력"""
        print("\n" + "="*60)
        print("📊 전처리 결과 요약")
        print("="*60)
        print(f"총 Q&A 개수: {len(self.qa_pairs)}")
        
        # 카테고리별 통계
        categories = {}
        for qa in self.qa_pairs:
            cat = qa['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\n카테고리별 분포:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {cat}: {count}개")
        
        # 답변 길이 통계
        lengths = [qa['metadata']['answer_length'] for qa in self.qa_pairs]
        print(f"\n답변 길이 통계:")
        print(f"  - 최소: {min(lengths)}자")
        print(f"  - 최대: {max(lengths)}자")
        print(f"  - 평균: {sum(lengths)/len(lengths):.1f}자")
        
        print("\n✅ 전처리 완료!")


def get_project_root() -> Path:
    """프로젝트 루트 경로 반환"""
    # 이 파일이 backend/ 폴더에 있으므로, parent.parent가 프로젝트 루트
    return Path(__file__).resolve().parent.parent.parent.parent


def main():
    """메인 실행 함수"""
    # 프로젝트 루트 경로 설정
    project_root = get_project_root()
    backend_dir = project_root / "backend"
    data_dir = backend_dir / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    # 출력 디렉토리 생성
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일 경로
    excel_path = raw_dir / "Q&A.xlsx"
    
    # 파일 존재 확인
    if not excel_path.exists():
        raise FileNotFoundError(
            f"❌ 파일을 찾을 수 없습니다: {excel_path}\n"
            f"   프로젝트 루트: {project_root}\n"
            f"   예상 위치: {excel_path}"
        )
    
    # 전처리 실행
    preprocessor = QAPreprocessor(str(excel_path))
    
    # 기본 전처리 (질문 변형 없이)
    qa_data = preprocessor.process(augment_questions=False)
    
    # 결과 저장
    json_path = processed_dir / "qa_preprocessed.json"
    csv_path = processed_dir / "qa_preprocessed.csv"
    
    preprocessor.save_to_json(str(json_path))
    preprocessor.save_to_csv(str(csv_path))
    
    # 요약 출력
    preprocessor.print_summary()
    
    # 샘플 데이터 출력
    print("\n" + "="*60)
    print("📝 샘플 데이터 (처음 2개)")
    print("="*60)
    for qa in qa_data[:2]:
        print(f"\nID: {qa['id']}")
        print(f"카테고리: {qa['category']}")
        print(f"질문: {qa['question']}")
        print(f"답변: {qa['answer'][:50]}...")
        print(f"키워드: {', '.join(qa['metadata']['keywords'])}")


if __name__ == '__main__':
    main()