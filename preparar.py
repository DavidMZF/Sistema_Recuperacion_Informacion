"""Genera la carpeta datos/ (corpus, consultas, qrels, índice invertido e índice FAISS) a partir de FiQA.

Uso:
    python descargar_datos.py          # una sola vez: baja FiQA a data/fiqa/
    python preparar.py                 # MUESTRA de 15.000 documentos (por defecto)
    python preparar.py --tamano 25000  # otra muestra
    python preparar.py --tamano 0      # corpus completo (necesita mucha RAM al desplegar)
    python analisis.py                 # recalcula datos/metricas.json
"""
import argparse
import glob

from ri import *

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="data/fiqa", help="carpeta con corpus/, queries/ y qrels/")
parser.add_argument("--tamano", type=int, default=15000,
                    help="nº de documentos de la muestra (0 = corpus completo)")
args = parser.parse_args()


def primero(patron):
    rutas = sorted(glob.glob(patron))
    if not rutas:
        raise SystemExit(f"No se encontró {patron}. Ejecuta antes: python descargar_datos.py")
    return rutas[0]


docs = cargar_corpus(primero(f"{args.base}/corpus/*.parquet"))
queries = cargar_queries(primero(f"{args.base}/queries/*.parquet"))
qrels = filtrar_qrels(docs, queries, cargar_qrels(primero(f"{args.base}/qrels/test.tsv")))
queries = {q: queries[q] for q in qrels}
print(f"Corpus original: {len(docs)} documentos, {len(qrels)} consultas con qrels")

if args.tamano > 0:
    docs = reducir_corpus(docs, qrels, args.tamano)
    qrels = filtrar_qrels(docs, queries, qrels)
    queries = {q: queries[q] for q in qrels}

idx = construir_indice(docs)
preparar_tfidf(idx)
modelo = Embedder()
index, ids = construir_faiss(docs, modelo)
guardar_sistema(docs, queries, qrels, index, ids, idx=idx)
print(f"Listo: {len(docs)} documentos, {len(qrels)} consultas con qrels guardados en datos/")
print("Siguiente paso: python analisis.py  (regenera datos/metricas.json)")
