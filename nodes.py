# pyrefly: ignore [missing-import]
from langchain_core.messages import AIMessage, SystemMessage

from llm import llm

from tools import tools


llm_with_tools = llm.bind_tools(tools)


SYSTEM_PROMPT = SystemMessage(
    content="""
Eres el asistente de BioWine.

Respondes preguntas en base a un conjunto de artículos científicos
seleccionados.

Reglas:
- Para cualquier pregunta sobre el contenido de los artículos, usa
  la herramienta buscar_en_articulos antes de responder.
- Basa tus respuestas en los fragmentos encontrados y cita la fuente
  (nombre del artículo y página).
- Si la información no aparece en los fragmentos, dilo claramente.
  No inventes datos.
- Los artículos pueden estar en inglés: responde siempre en el
  idioma del usuario.
- Para saludos o conversación informal no necesitas buscar.
"""
)


# Filtro de alcance: qué temas están dentro del alcance de BioWine y cuáles no.
ALCANCE = """
- El vino y su elaboración: vinificación, fermentación, crianza, calidad,
  defectos, características sensoriales.
- Viticultura: uvas, viñedos, variedades, terroir.
- Microbiología y microbioma del vino: levaduras, bacterias, hongos,
  interacciones microbianas, fagos.
- Bioquímica, biotecnología y métodos de estudio relacionados
  (ómicas, secuenciación, metabolómica, etc.).
- El contenido de las fuentes cargadas en BioWine.
- Lo que el usuario está viendo en pantalla dentro de BioWine.
- Preguntas sobre qué es BioWine o qué puede hacer.
- Saludos, agradecimientos y conversación informal breve.
- Seguimientos de la conversación (ej. "¿y eso qué significa?",
  "explícalo más simple", "dame un ejemplo").
"""

RESPUESTA_FUERA_DE_ALCANCE = (
    "Solo puedo ayudarte con temas relacionados con el vino, "
    "su microbiología y procesos de elaboración. "
    "¿Tienes alguna pregunta sobre eso?"
)


# ─────────────────────────────────────────────
# NODE 1 — Analizar alcance
# ─────────────────────────────────────────────

def formatear_historial(messages, max_messages=6, max_chars=300):

    # Solo mensajes del usuario y respuestas finales (sin tool calls)
    conversacion = [
        m for m in messages
        if m.type in ["human", "ai"] and m.content
    ]

    lineas = []

    for m in conversacion[-max_messages:]:

        rol = "USUARIO" if m.type == "human" else "BIOWINE"
        lineas.append(f"{rol}: {m.content[:max_chars]}")

    return "\n".join(lineas) or "(sin mensajes previos)"


def analizar_alcance(state):

    mensaje = state["messages"][-1].content

    historial = formatear_historial(state["messages"][:-1])

    prompt = f"""
Eres el filtro de alcance de BioWine, un asistente que responde
preguntas en base a artículos científicos sobre el vino.

Temas DENTRO del alcance:
{ALCANCE}

Está FUERA del alcance solo lo que claramente no tiene relación
razonable con esos temas. Por ejemplo:
"Enséñame a programar en Python."
"¿Quién ganó el partido de fútbol?"
"Escríbeme un poema sobre Napoleón."
"recomiendas una dosis de vino"

Usa la conversación previa para interpretar mensajes cortos o
de seguimiento.

No marques OUT_OF_SCOPE simplemente porque la pregunta no mencione
un tema específico de BioWine.

CONVERSACIÓN PREVIA:
{historial}

MENSAJE A CLASIFICAR:
{mensaje}

Responde con una sola palabra: DENTRO o FUERA
"""

    response = llm.invoke(prompt)

    content = response.content.upper()

    # Ante una respuesta ambigua, se deja pasar
    fuera = "FUERA" in content and "DENTRO" not in content

    print(f"\n[ALCANCE] {'FUERA' if fuera else 'DENTRO'}")

    return {
        "fuera_de_alcance": fuera
    }


# ─────────────────────────────────────────────
# NODE 2 — Fuera de alcance
# ─────────────────────────────────────────────

def responder_fuera_de_alcance(state):

    # Se agrega a messages para que quede en la conversación
    return {
        "messages": [
            AIMessage(content=RESPUESTA_FUERA_DE_ALCANCE)
        ]
    }


# ─────────────────────────────────────────────
# NODE 3 — Agente
# ─────────────────────────────────────────────

def agente(state):

    messages = [
        SYSTEM_PROMPT,
        *state["messages"],
    ]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }
