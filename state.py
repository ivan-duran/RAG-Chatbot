# pyrefly: ignore [missing-import]
from langgraph.graph import MessagesState

####################
# El State es la memoria de trabajo del proceso.
#
# MessagesState trae la lista "messages", que acumula la conversación:
# mensajes del usuario, respuestas del LLM, tool calls y sus resultados.
#
# Aquí se pueden agregar campos nuevos (ej. lo que se ve en pantalla).
####################

class BioWineState(MessagesState):

    fuera_de_alcance: bool
