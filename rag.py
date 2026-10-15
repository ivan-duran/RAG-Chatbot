import bisect
import logging
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path


# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma


####################
# RAG: artículos de referencia de BioWine.
#
# Uso:
# 1. Deja los artículos (.pdf, .txt, .md) en la carpeta "articulos/".
# 2. Ejecuta "python rag.py" para construir el índice en "chroma_db/".
#    Hay que reconstruirlo cada vez que cambian los artículos, el modelo
#    de embeddings o la forma de fragmentar.
# 3. La tool buscar_en_articulos (tools.py) usa obtener_retriever() para
#    buscar los fragmentos más relevantes para cada pregunta.
#
# Cómo se construye el índice (construir_indice):
# - PDF: se quita la bibliografía, se limpia cada página (encabezados,
#   ligaduras, palabras cortadas, saltos de línea) y se unen todas en un
#   solo texto, que se corta por oraciones en fragmentos de hasta 800
#   caracteres (unos 200 tokens de Qwen3).
# - .txt y .md: se cortan por párrafos y oraciones, sin limpieza.
# - Cada fragmento se convierte en un vector con Qwen3-Embedding y se
#   guarda en Chroma.
#
# Metadata de cada fragmento:
# - source: ruta del archivo.
# - start_index: posición del fragmento en el texto del archivo.
# - page, page_end (solo PDF): índice, desde 0, de la página donde el
#   fragmento empieza y termina. La tool los muestra sumando 1.
#
# Tests: tests/test_rag.py
####################

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "articulos"
DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "biowine_articulos"

# pypdf avisa por cada fuente que no reconoce; no afecta el texto extraído
logging.getLogger("pypdf").setLevel(logging.ERROR)


# ─────────────────────────────────────────────
# Limpieza y fragmentación
# ─────────────────────────────────────────────

# Encabezado de la bibliografía (se excluye del índice porque
# ensucia las búsquedas con listas de citas)
REFERENCES_HEADING = re.compile(
    r"^\s*(references|bibliography|literature cited|referencias|bibliografía)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Encabezados y pies de página ("CRITICAL REVIEWS IN ... 857",
# "858 Y. LIU ET AL."): líneas al borde de la página que, sin contar
# los números, se repiten en varias páginas del mismo artículo
LINEAS_BORDE = 3        # líneas que se revisan al inicio y al final
MIN_REPETICIONES = 3    # páginas en que debe aparecer la línea

# Palabra cortada al final de una línea: "inter-\nact" → "interact".
# Si sigue una mayúscula o un número se conserva el guion
# ("non-\nSaccharomyces" → "non-Saccharomyces").
# Una palabra compuesta cortada justo en su guion lo pierde
# ("quorum-\nsensing" → "quorumsensing"): sin un diccionario no se
# distingue de una palabra cortada.
PALABRA_CORTADA = re.compile(r"([^\W\d_])-\n([a-záéíóúüñ])")

# Fin de oración: punto tras minúscula, número o paréntesis, y luego
# mayúscula. Así no se corta en "S. cerevisiae" ni en "et al. (2014)"
FIN_DE_ORACION = r"(?<=[a-záéíóúüñ0-9)\]][.!?]) (?=[A-ZÁÉÍÓÚÑ¿¡])"

# Primero se corta por párrafos (.txt y .md), luego por oraciones y solo
# si una oración no cabe, por palabras. Los PDF no tienen saltos de
# línea después de limpiarlos: se cortan directamente por oraciones.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    # El solapamiento se arma con oraciones completas: con menos de ~200
    # casi ninguna cabe y los fragmentos quedan sin solapamiento
    chunk_overlap=200,
    separators=["\n\n", "\n", FIN_DE_ORACION, " ", ""],
    is_separator_regex=True,
    # El separador queda al final del trozo anterior: el punto con su oración
    keep_separator="end",
    # Guarda en la metadata la posición del fragmento en el texto;
    # con ella fragmentar_pdf calcula de qué página viene
    add_start_index=True,
)


# ─────────────────────────────────────────────
# Embeddings e índice
# ─────────────────────────────────────────────

# Embeddings locales y multilingües: permiten preguntar en español
# sobre artículos escritos en inglés.
# Qwen3-Embedding lee fragmentos de hasta 32k tokens (el modelo anterior
# los cortaba a los 128) y antepone a las consultas su instrucción de
# búsqueda ("query"); a los fragmentos no.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
    # Reemplaza a encode_kwargs en las consultas: se repite normalize
    query_encode_kwargs={"prompt_name": "query", "normalize_embeddings": True},
)

# Chroma rechaza más de 5461 vectores por llamada (unos 45 artículos):
# construir_indice() los agrega de a este número
FRAGMENTOS_POR_LOTE = 1000

# La base no se abre al importar este archivo: así construir_indice()
# puede borrar chroma_db/ completa sin que esté en uso.

def abrir_vectorstore():
    """Abre la colección de Chroma en chroma_db/ (la crea vacía si no existe)."""

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
        # Ambos se fijan al crear la colección (Chroma usa L2 por defecto)
        collection_configuration={"hnsw": {"space": "cosine"}},
        collection_metadata={"embedding_model": EMBEDDING_MODEL},
    )


def obtener_retriever():
    """Devuelve el retriever que usa la tool buscar_en_articulos.

    Cada búsqueda devuelve los 4 fragmentos más parecidos a la consulta.
    Lanza RuntimeError si el índice fue construido con otro modelo de
    embeddings.
    """

    vectorstore = abrir_vectorstore()

    # Un índice construido con otro modelo devuelve fragmentos sin
    # relación con la pregunta (o falla por la dimensión de los vectores)
    metadata = vectorstore._collection.metadata or {}

    if metadata.get("embedding_model") != EMBEDDING_MODEL:
        raise RuntimeError(
            f"El índice en {DB_DIR} no fue construido con {EMBEDDING_MODEL}. "
            "Reconstrúyelo con: python rag.py"
        )

    return vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )


# ─────────────────────────────────────────────
# Construcción del índice
# ─────────────────────────────────────────────

def quitar_referencias(pages):
    """Descarta la bibliografía de las páginas de un PDF.

    Devuelve las páginas hasta la que tiene el encabezado de la
    bibliografía, y esa página la recorta justo antes del encabezado
    (la modifica en el lugar). Si no lo encuentra, devuelve todas.
    """

    for i, page in enumerate(pages):

        match = REFERENCES_HEADING.search(page.page_content)

        if match:
            page.page_content = page.page_content[:match.start()]
            return pages[:i + 1]

    return pages


def normalizar_linea(linea):
    """Forma de una línea con la que se comparan los encabezados."""

    # Sin números: el número de página cambia en cada encabezado
    return re.sub(r"\d", "", linea).strip()


def encontrar_encabezados(pages):
    """Detecta los encabezados y pies de página de un artículo.

    Son las líneas que están entre las LINEAS_BORDE primeras o últimas
    de una página y se repiten en al menos MIN_REPETICIONES páginas.
    Devuelve un conjunto de líneas normalizadas con normalizar_linea.

    Una línea sin letras (por ejemplo, solo el número de página) no se
    toma como encabezado: al normalizarla quedaría vacía, y se borrarían
    también las líneas del texto que solo tienen números.
    """

    conteo = Counter()

    for page in pages:

        lineas = page.page_content.split("\n")
        borde = lineas[:LINEAS_BORDE] + lineas[-LINEAS_BORDE:]

        # Cada línea cuenta una sola vez por página
        conteo.update({normalizar_linea(linea) for linea in borde})

    return {
        linea for linea, veces in conteo.items()
        if veces >= MIN_REPETICIONES and re.search(r"[A-Za-z]", linea)
    }


def limpiar_pagina(texto, encabezados):
    """Limpia el texto de una página del PDF y lo deja en una sola línea.

    Quita los encabezados (los de encontrar_encabezados), reemplaza las
    ligaduras, une las palabras cortadas con guion y convierte los saltos
    de línea en espacios.

    pypdf a veces deja un espacio antes de una ligadura ("signi ﬁcant");
    ese espacio no se corrige y queda "signi ficant".
    """

    lineas = [
        linea for linea in texto.split("\n")
        if normalizar_linea(linea) not in encabezados
    ]

    texto = "\n".join(lineas)

    # Ligaduras tipográficas: "ﬁ" → "fi", "ﬂ" → "fl"
    texto = unicodedata.normalize("NFKC", texto)

    texto = PALABRA_CORTADA.sub(r"\1\2", texto)
    texto = texto.replace("-\n", "-")

    # Cada salto de línea del PDF es el fin de una línea de la columna,
    # no de un párrafo: se unen en un texto continuo
    return re.sub(r"\s+", " ", texto).strip()


def unir_paginas(pages):
    """Limpia las páginas de un PDF y las une en un solo texto.

    Devuelve (texto, inicios). inicios es una lista de pares
    (posición, página): la posición del texto en la que empieza cada
    página y su índice desde 0 (metadata "page" de PyPDFLoader).
    Las páginas que quedan vacías después de limpiarlas no aparecen.
    """

    encabezados = encontrar_encabezados(pages)

    texto = ""
    inicios = []    # (posición en el texto, índice de la página)

    for page in pages:

        contenido = limpiar_pagina(page.page_content, encabezados)

        if not contenido:
            continue

        # Palabra cortada entre una página y la siguiente
        if re.search(r"[^\W\d_]-$", texto) and contenido[0].islower():
            texto = texto[:-1]
        elif texto:
            texto += " "

        inicios.append((len(texto), page.metadata["page"]))
        texto += contenido

    return texto, inicios


def fragmentar_pdf(path):
    """Carga un PDF y lo divide en fragmentos listos para indexar.

    Cada fragmento lleva en su metadata source, start_index, page y
    page_end (ver el encabezado de este archivo).
    """

    pages = quitar_referencias(PyPDFLoader(str(path)).load())

    # Las páginas se unen antes de fragmentar para que un párrafo que
    # sigue en la página siguiente no quede cortado
    texto, inicios = unir_paginas(pages)

    chunks = splitter.create_documents([texto], [{"source": str(path)}])
    posiciones = [posicion for posicion, _ in inicios]

    def pagina_en(posicion):
        # Índice de la página que contiene esa posición del texto
        return inicios[bisect.bisect_right(posiciones, posicion) - 1][1]

    for chunk in chunks:

        inicio = chunk.metadata["start_index"]
        fin = inicio + len(chunk.page_content) - 1

        # Un fragmento puede seguir en la página siguiente: la tool
        # cita las dos ("páginas 3-4")
        chunk.metadata["page"] = pagina_en(inicio)
        chunk.metadata["page_end"] = pagina_en(fin)

    return chunks


def fragmentar_documentos():
    """Fragmenta todos los artículos de articulos/, también en subcarpetas.

    Los .pdf pasan por la limpieza; los .txt y .md se fragmentan tal
    cual. Los demás tipos de archivo se ignoran.
    """

    chunks = []

    for path in sorted(DOCS_DIR.rglob("*")):

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            chunks.extend(fragmentar_pdf(path))

        elif suffix in [".txt", ".md"]:
            docs = TextLoader(str(path), encoding="utf-8").load()
            chunks.extend(splitter.split_documents(docs))

    return chunks


def construir_indice():
    """Reconstruye chroma_db/ desde cero con los artículos de articulos/.

    Si no hay artículos, deja el índice que exista sin tocarlo.
    """

    chunks = fragmentar_documentos()

    if not chunks:
        print(f"No hay artículos en {DOCS_DIR}")
        return

    # Borramos la base completa y la creamos de cero: así no se duplican
    # fragmentos ni quedan carpetas de índices anteriores
    shutil.rmtree(DB_DIR, ignore_errors=True)

    vectorstore = abrir_vectorstore()

    for i in range(0, len(chunks), FRAGMENTOS_POR_LOTE):
        vectorstore.add_documents(chunks[i:i + FRAGMENTOS_POR_LOTE])

    archivos = len({chunk.metadata["source"] for chunk in chunks})

    print(f"✓ {archivos} archivos → {len(chunks)} fragmentos indexados")


if __name__ == "__main__":
    construir_indice()
