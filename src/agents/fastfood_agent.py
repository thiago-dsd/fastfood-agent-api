from datetime import datetime
from typing import Literal, Optional, cast
import logging
from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from core import get_model, settings

class AgentState(MessagesState, total=False):
    """Estado do agente para o FastFood."""
    order_details: str
    confirmed: bool  

def wrap_model(model: BaseChatModel, system_prompt: SystemMessage) -> RunnableSerializable[AgentState, AIMessage]:
    """Encapsula o modelo com um prompt de sistema."""
    preprocessor = RunnableLambda(
        lambda state: [system_prompt] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


order_capture_prompt = SystemMessagePromptTemplate.from_template("""
Você é um assistente de pedidos do FastFood. Sua tarefa é extrair os detalhes do pedido do usuário a partir do histórico de conversas.
Extraia os itens, quantidades e personalizações mencionados pelo usuário.
Se não houver detalhes suficientes, explique o que está faltando.
""")

confirmation_prompt = SystemMessagePromptTemplate.from_template("""
Você deve confirmar os detalhes do pedido com o usuário.
Se o usuário confirmar, o pedido será finalizado.
Se o usuário quiser corrigir algo, peça para ele fornecer os detalhes atualizados.
Seja paciente e educado!
""")

async def capture_order(state: AgentState, config: RunnableConfig) -> AgentState:
    """Este nó examina o histórico da conversa para determinar os detalhes do pedido como uma string.
    Se não houver detalhes suficientes, ele solicitará mais informações."""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m, order_capture_prompt.format())

    response = await model_runnable.ainvoke(state, config)

    order_details = response.content.strip()

    if not order_details or len(order_details.strip()) < 5:
        capture_input = interrupt(f"Parece que seu pedido está incompleto. Por favor, me informe mais detalhes.")
        state["messages"].append(HumanMessage(capture_input))
        return await capture_order(state, config)
    

    state["messages"].append(AIMessage("Detalhes do pedido capturados com sucesso!"))
    return {
        "confirmed": False,
        "order_details": order_details
    }

async def confirm_order(state: AgentState, config: RunnableConfig) -> AgentState:
    """Confirma os detalhes do pedido com o usuário."""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m, confirmation_prompt.format())

    confirm_input = interrupt(
        f"Seu pedido:\n{state.get('order_details')}\n\n"
        "Por favor, escolha uma opção:\n"
        "1) Confirmar pedido\n"
        "2) Alterar pedido"
    )
    state["messages"].append(HumanMessage(confirm_input))

    user_response = confirm_input.strip()
    if user_response == "1":
        state["messages"].append(AIMessage(interrupt("Pedido confirmado")))
        return {"confirmed": True}
    elif user_response == "2":
        state["messages"].append(AIMessage(interrupt("Certo, vamos alterar os pedidos!")))
        return {"confirmed": False}
    else:
        state["messages"].append(AIMessage(interrupt("Desculpe, não entendi. Por favor, escolha uma opção:\n1) Confirmar pedido\n2) Alterar pedido")))
        return {"confirmed": False}

async def register_order(state: AgentState, config: RunnableConfig) -> AgentState:
    """Registra o pedido no banco de dados e finaliza o fluxo."""
    order_id = register_order_in_db(state.get('order_details'))

    final_message = (
        f"✅ Pedido registrado com sucesso! Número do pedido: {order_id}\n"
        "Agradecemos sua preferência! Seu pedido já está a caminho. 🚚\n"
        "Se quiser fazer outro pedido, é só mandar uma mensagem aqui no chat!"
    )

    state["messages"].append(AIMessage(interrupt(final_message)))

    return

# Função auxiliar para registrar o pedido no banco de dados
def register_order_in_db(order_details: str) -> str:
    """Registra o pedido no banco de dados e retorna o ID."""
    # Implemente a lógica de registro aqui
    return "12345"  # Simula um ID de pedido


def check_confirmation(state: AgentState) -> Literal["register_order", "capture_order"]:
    """Verifica se o pedido foi confirmado e retorna o próximo nó."""
    if state.get("confirmed"):
        return "register_order"
    else:
        return "capture_order"

# Define o grafo do agente
agent = StateGraph(AgentState)
agent.add_node("capture_order", capture_order)
agent.add_node("confirm_order", confirm_order)
agent.add_node("register_order", register_order)

agent.set_entry_point("capture_order")

agent.add_edge("capture_order", "confirm_order")
agent.add_edge("confirm_order", "register_order")
agent.add_edge("register_order", END)

agent.add_conditional_edges(
    "confirm_order",
    check_confirmation,
    {"register_order": "register_order", "capture_order": "capture_order"},
)

# Compila o agente
fastfood_agent = agent.compile(checkpointer=MemorySaver())
fastfood_agent.name = "fastfood-agent"