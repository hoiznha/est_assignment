FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 포트 노출 (Railway가 PORT 환경변수를 제공)
EXPOSE $PORT

# Railway가 PORT 환경변수를 제공하므로 동적으로 사용
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}

