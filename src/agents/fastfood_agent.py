import logging

from datetime import datetime
from typing import Literal, cast

from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from core import get_model, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    """Este nó captura o primeiro pedido do usuário."""
    logging.info("CAPTURE ORDER")
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m, order_capture_prompt.format())

    response = await model_runnable.ainvoke(state, config)

    order_details = response.content.strip()    

    state["messages"].append(AIMessage("Detalhes do pedido capturados com sucesso!"))
    return {
        "confirmed": False,
        "order_details": order_details
    }

async def confirm_order(state: AgentState, config: RunnableConfig) -> AgentState:
    """Confirma os detalhes do pedido com o usuário. Dá a ele a opção de confirmar o pedido ou alterar."""
    logging.info("CONFIRM ORDER")
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m, confirmation_prompt.format())

    confirm_input = interrupt(
        f"Seu pedido:\n{state.get('order_details')}\n\n"
        "O que você deseja fazer?\n"
        "✅ Digite **'confirmar'** para finalizar o pedido.\n"
        "✏️ Digite **'alterar'** para modificar algo."
    )

    state["messages"].append(HumanMessage(confirm_input))

    user_response = confirm_input

    logging.info(f"user_response: {user_response}")
    
    if user_response.lower() in ["confirmar", "sim", "ok"]:
        return {"confirmed": True}
    elif user_response.lower() in ["alterar", "editar", "mudar"]:
        state["messages"].append(AIMessage("Certo! Vamos ajustar seu pedido. ✏️"))
        return {"confirmed": False}
    else:
        state["messages"].append(AIMessage("Desculpe, não entendi. Digite 'confirmar' ou 'alterar'."))
        return {"confirmed": False}


async def update_order(state: AgentState, config: RunnableConfig) -> AgentState:
    """Este nó atualiza os detalhes do pedido com base nas novas informações fornecidas pelo usuário."""
    logging.info("UPDATE ORDER")

    # Solicita ao usuário que escreva as alterações desejadas
    update_input = interrupt("Escreva as alterações que você queira realizar no pedido:")
    state["messages"].append(HumanMessage(update_input))

    # Captura a resposta do usuário
    user_response = update_input

    # Log da resposta do usuário
    logging.info(f"user_response: {user_response}")

    # Atualiza os detalhes do pedido com base na resposta do usuário
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m, order_capture_prompt.format())

    # Passa a resposta do usuário para o modelo processar
    response = await model_runnable.ainvoke(state, config)

    # Extrai os detalhes atualizados do pedido
    order_details = response.content.strip()

    # Adiciona uma mensagem de confirmação ao estado
    state["messages"].append(AIMessage("Detalhes do pedido atualizados com sucesso!"))

    # Retorna o estado atualizado
    return {
        "confirmed": False,
        "order_details": order_details
    }

async def register_order(state: AgentState, config: RunnableConfig) -> AgentState:
    """Agradece pelo pedido e finaliza o fluxo."""
    logging.info("REGISTER ORDER")

    final_message = (
        f"✅ Pedido registrado com sucesso!\n"
        "Agradecemos sua preferência! Seu pedido já está a caminho. 🚚\n"
        "Se quiser fazer outro pedido, é só mandar uma mensagem aqui no chat!"
    )

    logging.info("adicionando final message")
    state["messages"].append(AIMessage(
        content=final_message,
        response_metadata={
            "order_details": {
                "description": state.get("order_details"),
            }
        }
    ))

    return state

def check_confirmation(state: AgentState) -> Literal["register_order", "update_order"]:
    """Verifica se o pedido foi confirmado e retorna o próximo nó."""
    if state.get("confirmed"):
        return "register_order"
    else:
        return "update_order"

# Define o grafo do agente
agent = StateGraph(AgentState)
agent.add_node("capture_order", capture_order)
agent.add_node("confirm_order", confirm_order)
agent.add_node("update_order", update_order)
agent.add_node("register_order", register_order)

agent.set_entry_point("capture_order")

agent.add_edge("capture_order", "confirm_order")
agent.add_edge("update_order", "capture_order")
agent.add_edge("register_order", END)

agent.add_conditional_edges(
    "confirm_order",
    check_confirmation,
    {"register_order": "register_order", "update_order": "update_order"},
)

# Compila o agente
fastfood_agent = agent.compile(checkpointer=MemorySaver())
fastfood_agent.name = "fastfood-agent"