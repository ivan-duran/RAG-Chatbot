from pathlib import Path

# pyrefly: ignore [missing-import]
from langchain_core.tools import tool

from rag import obtener_retriever


retriever = obtener_retriever()


@tool
def buscar_en_articulos(query: str) -> str:
    """Busca fragmentos relevantes en los artículos científicos seleccionados
    de BioWine. Escribe consultas concretas y en inglés (el idioma de los
    artículos), aunque el usuario pregunte en español."""

    print(f"\n[RAG] Buscando en artículos: {query}")

    docs = retriever.invoke(query)

    if not docs:
        return "No se encontraron fragmentos relevantes en los artículos."

    fragmentos = []

    for doc in docs:

        fuente = Path(doc.metadata.get("source", "")).name
        pagina = doc.metadata.get("page")

        if pagina is not None:
            fuente += f", página {pagina + 1}"

        print(f"[RAG] ← {fuente}")

        fragmentos.append(
            f"[Fuente: {fuente}]\n{doc.page_content}"
        )

    return "\n\n".join(fragmentos)


# ─────────────────────────────────────────────
# Tools disponibles para el chatbot.
# Para agregar una nueva (ej. la de pantalla),
# defínela aquí y súmala a esta lista.
# ─────────────────────────────────────────────

tools = [
    buscar_en_articulos,
]
