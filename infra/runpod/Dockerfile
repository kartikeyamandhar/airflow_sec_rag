# RunPod serverless embedding worker (bge via fastembed).
# Build and push to a registry RunPod can pull, then create a serverless endpoint.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model into the image so cold starts do not download it.
ENV EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

COPY handler.py .
CMD ["python", "-u", "handler.py"]
