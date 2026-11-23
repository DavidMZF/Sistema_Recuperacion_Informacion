import json
import re
import time

from flask import Flask, request, render_template
from markupsafe import Markup, escape

from ri import *

app = Flask(__name__)
MODELOS = ["binario", "tfidf", "bm25", "embeddings"]
N_MIN, N_MAX = 1, 20

sis, _, _ = cargar_sistema("datos")
with open("datos/metricas.json", encoding="utf-8") as f:
    METRICAS = json.load(f)


def leer_n(valor, defecto=5):
    """Valida top-n en el servidor: entero entre N_MIN y N_MAX."""
    try:
        return max(N_MIN, min(N_MAX, int(valor)))
    except (TypeError, ValueError):
        return defecto


@app.template_filter("resaltar")
def resaltar(texto, consulta, limite=None):
    """Recorta el texto (en límite de palabra) y resalta las palabras de la consulta
    (comparando por raíz, igual que el índice). Devuelve HTML seguro."""
    tallos = {raiz(t) for t in re.findall(r"[a-z0-9]+", (consulta or "").lower())
              if t not in STOP and len(t) > 1}
    cortado = bool(limite) and len(texto) > limite
    if cortado:
        texto = texto[:limite].rsplit(" ", 1)[0]
    salida = []
    for parte in re.split(r"([A-Za-z0-9]+)", texto):
        palabra = parte.lower()
        if parte and re.fullmatch(r"[A-Za-z0-9]+", parte) and len(parte) > 1 \
                and palabra not in STOP and raiz(palabra) in tallos:
            salida.append(f"<mark>{escape(parte)}</mark>")
        else:
            salida.append(str(escape(parte)))
    return Markup("".join(salida) + ("…" if cortado else ""))


@app.route("/", methods=["GET", "POST"])
def inicio():
    consulta, resultados, tiempos, n = "", {}, {}, 5
    if request.method == "POST":
        consulta = request.form.get("consulta", "").strip()
        n = leer_n(request.form.get("n"))
        elegidos = set(request.form.getlist("modelos"))
        if consulta:
            for m in MODELOS:                          # orden fijo; ignora valores inventados
                if m in elegidos:
                    t0 = time.perf_counter()
                    resultados[m] = buscar(sis, m, consulta, n)
                    tiempos[m] = (time.perf_counter() - t0) * 1000
    return render_template("index.html", consulta=consulta, resultados=resultados,
                           tiempos=tiempos, n=n, modelos=MODELOS, metricas=METRICAS,
                           docs=sis["docs"])


if __name__ == "__main__":
    app.run(debug=False)