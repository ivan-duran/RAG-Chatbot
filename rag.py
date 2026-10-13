import logging
import re
import shutil
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
# 1. Deja los artículos (.pdf, .txt, .md) en la carpeta "articulos/".
# 2. Ejecuta "python rag.py" para construir el índice en "chroma_db/".
# 3. La tool buscar_en_articulos usa el retriever para buscar
#    los fragmentos más relevantes para cada pregunta.
####################

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "articulos"
DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "biowine_articulos"

# pypdf avisa por cada fuente que no reconoce; no afecta el texto extraído
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Encabezado de la bibliografía (se excluye del índice porque
# ensucia las búsquedas con listas de citas)
REFERENCES_HEADING = re.compile(
    r"^\s*(references|bibliography|literature cited|referencias|bibliografía)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Embeddings locales y multilingües: permiten preguntar en español
# sobre artículos escritos en inglés.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# La base no se abre al importar este archivo: así construir_indice()
# puede borrar chroma_db/ completa sin que esté en uso.

def abrir_vectorstore():

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )


def obtener_retriever():

    return abrir_vectorstore().as_retriever(
        search_kwargs={"k": 4}
    )


# ─────────────────────────────────────────────
# Construcción del índice
# ─────────────────────────────────────────────

def quitar_referencias(pages):

    for i, page in enumerate(pages):

        match = REFERENCES_HEADING.search(page.page_content)

        if match:
            page.page_content = page.page_content[:match.start()]
            return pages[:i + 1]

    return pages


def cargar_documentos():

    docs = []

    for path in sorted(DOCS_DIR.rglob("*")):

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            pages = PyPDFLoader(str(path)).load()
            docs.extend(quitar_referencias(pages))

        elif suffix in [".txt", ".md"]:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())

    return docs


def construir_indice():

    docs = cargar_documentos()

    if not docs:
        print(f"No hay artículos en {DOCS_DIR}")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(docs)

    # Borramos la base completa y la creamos de cero: así no se duplican
    # fragmentos ni quedan carpetas de índices anteriores
    shutil.rmtree(DB_DIR, ignore_errors=True)

    vectorstore = abrir_vectorstore()
    vectorstore.add_documents(chunks)

    print(f"✓ {len(docs)} páginas → {len(chunks)} fragmentos indexados")


if __name__ == "__main__":
    construir_indice()
