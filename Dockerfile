FROM python:3.11-slim as builder

WORKDIR /app

# 빌드에 필요한 시스템 의존성만 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 복사 및 설치
# PyTorch CPU-only 버전을 먼저 설치 (CUDA 제거로 이미지 크기 대폭 감소)
# Railway는 CPU만 사용하므로 CPU-only 버전 사용
COPY requirements.txt .
RUN pip install --no-cache-dir --user \
    torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r requirements.txt

# 최종 이미지 (경량화)
FROM python:3.11-slim

WORKDIR /app

# 런타임에는 추가 패키지 불필요 (python:3.11-slim에 기본 포함)

# 빌드 스테이지에서 설치한 패키지 복사
COPY --from=builder /root/.local /root/.local

# PATH에 사용자 설치 패키지 추가
ENV PATH=/root/.local/bin:$PATH

# 애플리케이션 코드 복사
COPY . .

# 포트 노출 (Railway가 PORT 환경변수를 제공)
EXPOSE $PORT

# Railway가 PORT 환경변수를 제공하므로 동적으로 사용
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}

