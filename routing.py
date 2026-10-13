def route_scope(state):

    if state["fuera_de_alcance"]:
        return "fuera_de_alcance"

    return "dentro"


def route_after_llm(state):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return "response"
