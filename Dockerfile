# ---- Base Stage ----
FROM python:3.11-slim AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Create a non-root user and group
RUN addgroup --system nonroot && adduser --system --ingroup nonroot nonroot

# ---- Builder Stage (Optional but good practice if you have build-time deps) ----
# FROM base AS builder
# WORKDIR /app
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev # Example build deps
# COPY requirements.txt .
# RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ---- Final Stage ----
FROM base AS final

WORKDIR /app

# Copy dependencies first to leverage Docker cache
COPY requirements.txt .
# Optional: If using builder stage with wheels:
# COPY --from=builder /wheels /wheels
# RUN pip install --no-cache --no-index --find-links=/wheels -r requirements.txt && rm -rf /wheels
# Standard install:
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R nonroot:nonroot /app

# Switch to non-root user
USER nonroot

# Expose the port the app runs on (optional but good documentation)
EXPOSE 8080

# Run the application - Gunicorn will respect PORT env var (default 8080 in Cloud Run)
# Let Cloud Run/Gunicorn determine workers via WEB_CONCURRENCY if possible
CMD ["gunicorn", "main:app"]
# Alternatively, be explicit about binding to $PORT if needed/preferred:
# CMD ["gunicorn", "-b", "0.0.0.0:${PORT}", "main:app"]
