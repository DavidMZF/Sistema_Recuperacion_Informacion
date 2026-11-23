# Sistema de Recuperación de Información (FiQA-2018)

Proyecto de 1er Bimestre — Recuperación de Información (EPN). Compara cuatro modelos de recuperación sobre una **muestra del corpus FiQA-2018** (BEIR, preguntas y respuestas financieras en inglés) con una interfaz web. La muestra (15.000 documentos por defecto) conserva todos los documentos relevantes de las consultas de prueba y completa con documentos al azar (semilla fija); así la aplicación cabe en un servicio gratuito de 512 MB.

**URL del despliegue:** _(pendiente — completar al desplegar; ver `GUIA_DESPLIEGUE_RENDER.md`)_

## Modelos

| Modelo | Implementación |
|---|---|
| Binario | Similitud **Jaccard** sobre vectores binarios de términos (propia, sobre el índice invertido) |
| Vectorial TF-IDF | **Coseno** con TF-IDF (tf = 1 + log f; idf = log(1 + N/df)) (propia) |
| Probabilístico | **BM25** (k1 = 1.5, b = 0.75) (propia) |
| Vectorial con embeddings | `all-MiniLM-L6-v2` en formato ONNX (fastembed, sin PyTorch) + búsqueda en **FAISS** (`IndexFlatIP`, vectores normalizados = coseno) |

Procesamiento de texto: minúsculas, tokenización por regex (`[a-z0-9]+`), stopwords en inglés (lista de NLTK guardada en `datos/stopwords.json`) y stemming Porter. El **índice invertido** guarda, por término, los documentos donde aparece y su frecuencia. No se usa ningún motor de búsqueda (Elasticsearch, Solr, Whoosh).

Evaluación: las consultas de prueba de FiQA (`test.tsv`) que tienen qrels. Por consulta: Precision, Recall (y F1, NDCG) en k = 5, 10 y 20. Para todo el sistema: **MAP** (calculado sobre los 100 primeros resultados, es decir MAP@100), MRR y NDCG.

## Estructura

```
ri.py                 # preprocesamiento, índice invertido, modelos, métricas
app.py                # aplicación web (Flask)
templates/index.html  # interfaz: buscar/comparar y evaluación
analisis.py           # evalúa los 4 modelos y genera datos/metricas.json
preparar.py           # genera datos/ (muestra del corpus, qrels, índice invertido, índice FAISS)
descargar_datos.py    # baja FiQA a data/fiqa/
descargar_modelo.py   # baja el modelo de embeddings (se usa en el build del despliegue)
render.yaml           # configuración de despliegue en Render
datos/                # muestra procesada + índices + métricas (ya incluidos)
requirements.txt      # dependencias para ejecutar la app
requirements-datos.txt# dependencias extra para regenerar datos/
```

## Ejecutar la aplicación

La carpeta `datos/` ya está incluida, así que no hace falta descargar nada. Se recomienda Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                      # http://127.0.0.1:5000
```

La primera ejecución descarga el modelo de embeddings (formato ONNX) a `modelo_cache/`. El índice invertido ya viene precalculado en `datos/indice.pkl`, por lo que el arranque es rápido.

Con servidor de producción: `gunicorn app:app --workers 1 --timeout 120`.

## Regenerar los datos y las métricas (opcional)

```bash
pip install -r requirements-datos.txt
python descargar_datos.py          # FiQA (BeIR/fiqa y BeIR/fiqa-qrels, Hugging Face) -> data/fiqa/
python preparar.py                 # muestra de 15.000 documentos; otras opciones: --tamano 25000, --tamano 0 (corpus completo)
python analisis.py                 # imprime el resumen y actualiza datos/metricas.json
```

`--tamano N` fija el tamaño de la muestra, conservando **todos** los relevantes de los qrels y completando con documentos al azar (semilla fija). Las métricas se calculan sobre la muestra, por lo que no son comparables con las publicadas para el corpus completo de FiQA.

## Uso de la interfaz

- **Buscar y comparar:** escribe una consulta, elige uno o varios modelos y el top-n (1–20). Con varios modelos se muestran en columnas; los puntos de color de cada documento indican en qué otros modelos aparece.
- **Evaluación:** MAP, MRR, P@k, R@k, F1@k y NDCG@k por modelo; comparación de embeddings frente a BM25/TF-IDF; resultados por tipo de consulta, ejemplos y detalle por consulta.

## Limitaciones

- El corpus y el modelo de embeddings están en **inglés**: las consultas deben escribirse en inglés (en español, los embeddings rinden mal).
- El modelo de embeddings trunca los textos largos (límite de longitud del modelo); solo se embebe la parte inicial.
- El corpus se carga desde archivos Parquet de BEIR y se guarda como JSON de texto plano (`datos/docs.json`).
- MAP se calcula sobre los 100 primeros resultados (MAP@100).
- Se evalúa sobre una muestra del corpus (con menos distractores que el corpus completo), así que las métricas absolutas salen más altas.
