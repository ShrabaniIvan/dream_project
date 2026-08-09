FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (without building the project package itself)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source, data, and readme (readme required by hatchling)
COPY README.md ./
COPY src/ ./src/
COPY data/ ./data/

# Install the project package
RUN uv sync --frozen --no-dev

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "src/dream_project/app.py", "--server.address=0.0.0.0"]
