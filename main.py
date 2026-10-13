from uuid import uuid4

# pyrefly: ignore [missing-import]
from langchain_core.messages import HumanMessage, RemoveMessage

from graph import graph


def main():

    print("=" * 60)
    print("              BioWine")
    print("       Asistente de artículos")
    print("=" * 60)

    print("\nEscribe 'salir' para terminar.")
    print("-" * 60)

    # Un thread por sesión: el checkpointer guarda aquí la conversación
    config = {
        "configurable": {"thread_id": str(uuid4())}
    }

    while True:

        user_input = input("\nTú: ")

        if user_input.lower() in [
            "salir",
            "exit",
            "quit"
        ]:
            break

        if not user_input.strip():
            continue
        
        # antes de llamar al grafo, guardamos cuántos mensajes hay
        num_mensajes = len(graph.get_state(config).values.get("messages", []))

        try:

            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config,
            )

            print(
                "\nBioWine:",
                result["messages"][-1].content
            )

        except Exception as e:
            nuevos = graph.get_state(config).values["messages"][num_mensajes:]
            graph.update_state(
                config,
                {"messages": [RemoveMessage(id=m.id) for m in nuevos]}
            )
            print(
                f"\n[ERROR] {e}"
            )


if __name__ == "__main__":
    main()
