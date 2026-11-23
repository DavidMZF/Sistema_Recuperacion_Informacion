"""Descarga el modelo de embeddings (ONNX) a modelo_cache/. Se ejecuta en el build del despliegue."""
from ri import Embedder

Embedder().encode(["warm-up"])
print("Modelo de embeddings listo.")
