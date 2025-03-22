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
    capture_order_state: Literal["order", "end", "conversation"]
    confirm_order_state: Literal["confirmed", "redirect_to_capture"]

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
    """Este nó captura o pedido do usuário ou redireciona o fluxo com base na intenção do usuário."""
    logging.info("CAPTURE ORDER")
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))

    # Verifica se o usuário está interessado em fazer um pedido ou apenas conversando
    intent_prompt = SystemMessagePromptTemplate.from_template("""
    Analise a conversa do usuário e determine se ele está interessado em fazer um pedido ou apenas conversando.
    Responda com:
    - "order" se o usuário estiver interessado em fazer um pedido.
    - "conversation" se o usuário estiver falando sobre assuntos que não envolvam pedido ou
    estiver solicitando informações sobre cardápio.
    - "end" se o usuário quiser finalizar a interação.
    Histórico de conversa: {messages}
    """)

    intent_model = wrap_model(m, intent_prompt.format(messages=state["messages"]))
    intent_response = await intent_model.ainvoke(state, config)
    intent = intent_response.content.strip().lower()

    logging.info(f"Intenção detectada: {intent}")

    if intent == "order":
        order_capture_model = wrap_model(m, order_capture_prompt.format())
        response = await order_capture_model.ainvoke(state, config)
        order_details = response.content.strip()

        state["messages"].append(AIMessage("Detalhes do pedido capturados com sucesso!"))
        return {
            "order_details": order_details,
            "capture_order_state": "order",
        }
    elif intent == "end":
        state["messages"].append(AIMessage("Obrigado por interagir conosco! Até a próxima. 👋"))
        return {
            "capture_order_state": "end",
        }
    else:
        state["messages"].append(AIMessage("Como posso ajudar você hoje?"))
        return {
            "capture_order_state": "conversation",
        }
    
async def handle_conversation(state: AgentState, config: RunnableConfig) -> AgentState:
    """Este nó lida com conversas genéricas, como perguntas sobre o cardápio ou informações gerais."""
    logging.info("HANDLE CONVERSATION")
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))

    # Pergunta ao usuário sobre o que ele gostaria de saber
    conversation_prompt = SystemMessagePromptTemplate.from_template("""
    Você é um assistente de atendimento ao cliente do FastFood.
    Responda às perguntas do usuário de forma educada e útil.
    Se o usuário quiser fazer um pedido, peça para ele fornecer os detalhes.
    Histórico de conversa: {messages}
    """)

    conversation_model = wrap_model(m, conversation_prompt.format(messages=state["messages"]))
    response = await conversation_model.ainvoke(state, config)
    response_message = response.content.strip()

    state["messages"].append(AIMessage(response_message))
    return state

async def confirm_order(state: AgentState, config: RunnableConfig) -> AgentState:
    """Confirma os detalhes do pedido com o usuário. Interpreta a intenção do usuário."""
    logging.info("CONFIRM ORDER")
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m, confirmation_prompt.format())

    # Pergunta ao usuário de forma mais natural
    confirm_input = interrupt(
        f"Seu pedido:\n{state.get('order_details')}\n\n"
        "Por favor, confirme se está tudo certo ou se deseja fazer alguma alteração."
    )
    state["messages"].append(HumanMessage(confirm_input))

    # Captura a resposta do usuário
    user_response = confirm_input

    logging.info(f"user_response: {user_response}")

    # Usa o modelo de linguagem para interpretar a intenção do usuário
    intent_prompt = SystemMessagePromptTemplate.from_template("""
    Faça uma análise da RespostaBase do usuário e retorne:
    - "confirm" caso ele diga de forma explícita que deseja confirmar/finalizar.
    - "change" caso ele comente sobre adicionar novos itens ou alterar os itens já existentes.
    RespostaBase: {user_response}
    """)
    intent_model = wrap_model(m, intent_prompt.format(user_response=user_response))
    intent_response = await intent_model.ainvoke(state, config)
    intent = intent_response.content.strip().lower()

    logging.info(f"Intenção detectada: {intent}")

    # Decide o próximo passo com base na intenção detectada
    if intent == "confirm":
        state["messages"].append(AIMessage("Pedido confirmado! ✅"))
        return {"confirm_order_state": "confirmed"}  # Atualizado
    elif intent == "change":
        state["messages"].append(AIMessage("Certo! Vamos ajustar seu pedido. ✏️"))
        return {"confirm_order_state": "redirect_to_capture"}  # Atualizado
    else:
        state["messages"].append(AIMessage("Desculpe, não entendi. Poderia repetir, por favor?"))
        return {"confirm_order_state": None}  # Ou outro valor padrão, se necessário

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

def check_confirmation(state: AgentState) -> Literal["register_order", "capture_order", "confirm_order"]:
    """
    Verifica o estado atual e decide o próximo nó.
    - Se o pedido foi confirmado, redireciona para "register_order".
    - Se o usuário quer alterar o pedido, redireciona para "capture_order".
    - Caso contrário, repete o nó "confirm_order".
    """
    if state.get("confirm_order_state") == "confirmed":
        return "register_order"
    elif state.get("confirm_order_state") == "redirect_to_capture":
        return "capture_order"
    else:
        return "confirm_order"

def check_capture(state: AgentState) -> Literal["confirm_order", "handle_conversation", "END"]:
    """
    Verifica o estado atual e decide o próximo nó.
    - Se o usuário estiver fazendo um pedido, redireciona para "confirm_order".
    - Se o usuário estiver apenas conversando, redireciona para "handle_conversation".
    - Se o usuário quiser finalizar, redireciona para "END".
    """
    if state.get("capture_order_state") == "order":
        return "confirm_order"
    elif state.get("capture_order_state") == "conversation":
        return "handle_conversation"
    elif state.get("capture_order_state") == "end":
        return END

agent = StateGraph(AgentState)
agent.add_node("capture_order", capture_order)
agent.add_node("handle_conversation", handle_conversation)
agent.add_node("confirm_order", confirm_order)
agent.add_node("register_order", register_order)

agent.set_entry_point("capture_order")

agent.add_edge("register_order", END)
# agent.add_edge("handle_conversation", "capture_order")

agent.add_conditional_edges(
    "capture_order",
    check_capture,
    {
        "confirm_order": "confirm_order", 
        "handle_conversation": "handle_conversation",
        END: END
    },
)

agent.add_conditional_edges(
    "confirm_order",
    check_confirmation,
    {
        "register_order": "register_order",
        "capture_order": "capture_order",
        "confirm_order": "confirm_order", 
    },
)


fastfood_agent = agent.compile(checkpointer=MemorySaver())
fastfood_agent.name = "fastfood-agent"