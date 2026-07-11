# Model-serving workload for the ML + robotics fleet.
# The candidate and baseline both run this image, parameterized by MODEL_VERSION.
FROM python:3.11-slim

# Prometheus scrape port for the health signals the decision tree gates on.
EXPOSE 8080

WORKDIR /app

# Serving deps would go here (e.g. the inference runtime). Kept minimal so the
# image builds in CI; the exposition format is what the deployment engine reads.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
ENV PYTHONPATH=/app/src

# MODEL_VERSION selects which weights this replica serves; the K8s manifests set it
# per color (blue = baseline, green = candidate).
ENV MODEL_VERSION="baseline"
ENV METRICS_PORT="8080"

# In a real deployment this would launch the inference server exposing /metrics.
# Here it prints the model version so the container has a valid, buildable entrypoint.
CMD ["python", "-c", "import os; print(f'serving model {os.environ.get(\"MODEL_VERSION\")} on :{os.environ.get(\"METRICS_PORT\")}')"]
