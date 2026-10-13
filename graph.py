# pyrefly: ignore [missing-import]
from langgraph.graph import (
    StateGraph,
    START,
    END,
)

# pyrefly: ignore [missing-import]
from langgraph.prebuilt import ToolNode
# pyrefly: ignore [missing-import]
from langgraph.checkpoint.memory import InMemorySaver

from state import BioWineState

from nodes import (
    analizar_alcance,
    responder_fuera_de_alcance,
    agente,
)

from routing import (
    route_scope,
    route_after_llm,
)

from tools import tools


# ─────────────────────────────────────────────
# Graph
#
# START → analizar_alcance
#           ├─ fuera_de_alcance → responder_fuera_de_alcance → END
#           └─ dentro → agente ⇄ tools
#                         └─ sin tool_calls → END
# ─────────────────────────────────────────────

builder = StateGraph(BioWineState)


# Nodes

builder.add_node(
    "analizar_alcance",
    analizar_alcance
)

builder.add_node(
    "fuera_de_alcance",
    responder_fuera_de_alcance
)

builder.add_node(
    "agente",
    agente
)

builder.add_node(
    "tools",
    ToolNode(tools)
)


# Edges

builder.add_edge(
    START,
    "analizar_alcance"
)


# Scope routing

builder.add_conditional_edges(
    "analizar_alcance",
    route_scope,
    {
        "fuera_de_alcance": "fuera_de_alcance",
        "dentro": "agente",
    }
)

builder.add_edge(
    "fuera_de_alcance",
    END
)


# Tool loop

builder.add_conditional_edges(
    "agente",
    route_after_llm,
    {
        "tools": "tools",
        "response": END,
    }
)

builder.add_edge(
    "tools",
    "agente"
)


# El checkpointer guarda la conversación entre turnos
# (en memoria: se pierde al cerrar el programa).

graph = builder.compile(
    checkpointer=InMemorySaver()
)
