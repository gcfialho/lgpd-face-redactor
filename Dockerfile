FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy LGPD_REDACTOR_MODELS=/models
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv sync --extra web --no-dev \
    # Bake the checksum-verified weights into the image so the container works offline.
    && uv run python -c "from lgpd_face_redactor import models; models.ensure(models.YUNET); models.ensure(models.SFACE)"

RUN useradd --create-home redactor && chown -R redactor /models
USER redactor
EXPOSE 8000
# Bind to all interfaces inside the container; publish it on localhost only (see README).
CMD ["uv", "run", "--no-sync", "lgpd-redact-web", "--host", "0.0.0.0", "--port", "8000"]
