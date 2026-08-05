# Single stage. Every dependency in requirements.txt must have a manylinux wheel,
# no compiler toolchain is installed.
FROM python:3.11-slim

WORKDIR /app

# Dependency layer cached separately from code, so code changes do not invalidate it.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY artifacts/ ./artifacts/
COPY sql/ ./sql/

ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Never hardcode 8080, Cloud Run injects PORT.
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
