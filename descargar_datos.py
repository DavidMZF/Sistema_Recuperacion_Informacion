"""Descarga FiQA-2018 (BEIR) desde Hugging Face a data/fiqa/ (corpus, queries, qrels)."""
import os
import shutil

from huggingface_hub import hf_hub_download, list_repo_files

DESTINO = "data/fiqa"


def bajar(repo, archivo, carpeta):
    os.makedirs(f"{DESTINO}/{carpeta}", exist_ok=True)
    ruta = hf_hub_download(repo_id=repo, filename=archivo, repo_type="dataset")
    destino = f"{DESTINO}/{carpeta}/{os.path.basename(archivo)}"
    shutil.copy(ruta, destino)
    print("OK", destino)


archivos = list_repo_files("BeIR/fiqa", repo_type="dataset")
for carpeta in ("corpus", "queries"):
    pq = [a for a in archivos if a.startswith(f"{carpeta}/") and a.endswith(".parquet")]
    if not pq:
        raise SystemExit(f"No hay parquet de '{carpeta}' en BeIR/fiqa; descárgalo a mano en {DESTINO}/{carpeta}/")
    bajar("BeIR/fiqa", pq[0], carpeta)

bajar("BeIR/fiqa-qrels", "test.tsv", "qrels")
