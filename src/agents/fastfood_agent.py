from datetime import datetime
from typing import Optional, cast

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
    order_details: Optional[dict]  # Detalhes do pedido
    confirmed: bool  # Indica se o pedido foi confirmado


def wrap_model(model: BaseChatModel, system_prompt: SystemMessage) -> RunnableSerializable[AgentState, AIMessage]:
    """Encapsula o modelo com um prompt de sistema."""
    preprocessor = RunnableLambda(
        lambda state: [system_prompt] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


# Prompts do sistema
order_capture_prompt = SystemMessagePromptTemplate.from_template("""
Você é um assistente de pedidos do FastFood. Sua tarefa é ajudar o usuário a fazer seu pedido de forma rápida e fácil.
Pergunte ao usuário o que ele gostaria de pedir e confirme os detalhes antes de finalizar.
Seja claro e amigável!
""")

confirmation_prompt = SystemMessagePromptTemplate.from_template("""
Você deve confirmar os detalhes do pedido com o usuário.
Se o usuário confirmar, o pedido será finalizado.
Se o usuário quiser corrigir algo, peça para ele fornecer os detalhes atualizados.
Seja paciente e educado!
""")


async def capture_order(state: AgentState, config: RunnableConfig) -> AgentState:
    """Captura os detalhes do pedido do usuário."""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m, order_capture_prompt.format())
    
    # Se não houver pedido capturado ainda, pergunte ao usuário
    if not state.get("order_details"):
        response = await model_runnable.ainvoke({"messages": [HumanMessage(content="Olá! O que você gostaria de pedir hoje?")]}, config)
    else:
        response = await model_runnable.ainvoke(state, config)

    # Extrai os detalhes do pedido da resposta do modelo
    order_details = extract_order_details(response.content)
    return {"order_details": order_details, "messages": [response]}


async def confirm_order(state: AgentState, config: RunnableConfig) -> AgentState:
    """Confirma os detalhes do pedido com o usuário."""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m, confirmation_prompt.format())

    # Resume o pedido para o usuário
    order_summary = format_order_summary(state["order_details"])
    response = await model_runnable.ainvoke(
        {"messages": [HumanMessage(content=f"Por favor, confirme seu pedido:\n{order_summary}\n\nEstá correto? (sim/não)")]},
        config,
    )

    # Verifica se o usuário confirmou o pedido
    if "sim" in response.content.lower():
        return {"confirmed": True, "messages": [response]}
    else:
        return {"confirmed": False, "messages": [response]}


async def register_order(state: AgentState, config: RunnableConfig) -> AgentState:
    """Registra o pedido no banco de dados."""
    if not state.get("confirmed"):
        raise ValueError("Pedido não confirmado.")

    # Simula o registro no banco de dados
    order_id = register_order_in_db(state["order_details"])
    return {"messages": [AIMessage(content=f"✅ Pedido registrado com sucesso! Número do pedido: {order_id}\nAgradecemos sua preferência!")]}


# Função auxiliar para extrair detalhes do pedido
def extract_order_details(text: str) -> dict:
    """Extrai os detalhes do pedido do texto."""
    # Implemente a lógica de extração aqui (ex.: usar regex ou um modelo de NLP)
    return {"items": [], "quantities": [], "customizations": []}


# Função auxiliar para formatar o resumo do pedido
def format_order_summary(order_details: dict) -> str:
    """Formata os detalhes do pedido para exibição ao usuário."""
    items = order_details.get("items", [])
    quantities = order_details.get("quantities", [])
    customizations = order_details.get("customizations", [])

    summary = "Seu pedido:\n"
    for item, quantity in zip(items, quantities):
        summary += f"- {quantity}x {item}\n"
    if customizations:
        summary += "Personalizações:\n"
        for customization in customizations:
            summary += f"- {customization}\n"
    return summary


# Função auxiliar para registrar o pedido no banco de dados
def register_order_in_db(order_details: dict) -> str:
    """Registra o pedido no banco de dados e retorna o ID."""
    # Implemente a lógica de registro aqui
    return "12345"  # Simula um ID de pedido


# Define o grafo do agente
agent = StateGraph(AgentState)
agent.add_node("capture_order", capture_order)
agent.add_node("confirm_order", confirm_order)
agent.add_node("register_order", register_order)

agent.set_entry_point("capture_order")
agent.add_edge("capture_order", "confirm_order")
agent.add_conditional_edges(
    "confirm_order",
    lambda state: "register_order" if state.get("confirmed") else "capture_order",
    {"register_order": "register_order", "capture_order": "capture_order"},
)
agent.add_edge("register_order", END)

# Compila o agente
fastfood_agent = agent.compile(checkpointer=MemorySaver())
fastfood_agent.name = "fastfood-agent"