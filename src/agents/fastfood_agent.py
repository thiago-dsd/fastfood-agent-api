import logging
from typing import Literal

from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from pydantic import BaseModel, Field

from core import get_model, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AgentState(MessagesState, total=False):
    """Estado do agente para o FastFood."""
    order_details: str
    user_response: str
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
    """Este nó captura o pedido do usuário e verifica se os detalhes são suficientes."""
    logging.info("CAPTURE ORDER")
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))

    # Extrai os detalhes do pedido do histórico de conversas
    order_capture_prompt = SystemMessagePromptTemplate.from_template("""
    Analise o histórico de conversas abaixo e extraia os detalhes do pedido do usuário.
    Inclua:
    - Quantidade de cada item.
    - Sabores ou personalizações.
    - Tamanho (pequena, média, grande).
    - Acompanhamentos ou bebidas (se mencionados).

    Histórico de conversas:
    {messages}

    Detalhes do pedido:
    """)

    order_capture_model = wrap_model(m, order_capture_prompt.format(messages=state["messages"]))
    response = await order_capture_model.ainvoke(state, config)
    order_details = response.content.strip()

    logging.info(f"Detalhes do pedido capturados: {order_details}")

    # Verifica se os detalhes são suficientes
    validation_prompt = SystemMessagePromptTemplate.from_template("""
    Analise os detalhes do pedido abaixo e verifique se as informações fornecidas pelo usuário estão completas.
    Um pedido completo deve incluir pelo menos um dos seguintes dados:
    1. Quantidade de itens (ex: "2 pizzas", "1 refrigerante").
    2. Sabores ou personalizações (ex: "calabresa", "quatro queijos", "borda recheada").
    3. Tamanho (ex: "pequena", "média", "grande").

    Responda com:
    - "complete" se pelo menos um dos dados acima estiver presente.
    - "incomplete" se nenhum dos dados acima for mencionado.

    Exemplos:
    1. Pedido: "Quero uma pizza de calabresa, tamanho médio."
    Resposta: "complete" (contém sabor e tamanho).

    2. Pedido: "Gostaria de 2 refrigerantes."
    Resposta: "complete" (contém quantidade).

    3. Pedido: "Olá, quero fazer um pedido."
    Resposta: "incomplete" (não menciona quantidade, sabor ou tamanho).

    Detalhes do pedido: {order_details}
    """)
    
    validation_model = wrap_model(m, validation_prompt.format(order_details=order_details))
    validation_response = await validation_model.ainvoke(state, config)
    validation_result = validation_response.content.strip().lower()

    logging.info(f"Resultado da validação: {validation_result}")

    if validation_result == "complete":
        # Se os detalhes forem suficientes, avança para a confirmação
        state["order_details"] = order_details
        state["capture_order_state"] = "order"
        state["messages"].append(AIMessage("Detalhes do pedido capturados com sucesso!"))
    else:
        # Se faltarem detalhes, solicita mais informações ao usuário
        state["messages"].append(AIMessage("Parece que faltam alguns detalhes no seu pedido. Por favor, forneça mais informações."))
        state["capture_order_state"] = "conversation"  # Volta para o modo de conversa

    return state
    
async def handle_conversation(state: AgentState, config: RunnableConfig) -> AgentState:
    """Este nó lida com conversas genéricas, como perguntas sobre o cardápio ou informações gerais."""
    logging.info("HANDLE CONVERSATION")
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))

    # Pergunta ao usuário sobre o que ele gostaria de saber
    conversation_prompt = SystemMessagePromptTemplate.from_template("""
    Você é um assistente de atendimento ao cliente do FastFood.
    Responda às perguntas do usuário de forma educada e útil.
    Se o usuário quiser fazer um pedido, peça para ele fornecer os detalhes.
    Responsa possíveis dúvidas dele.
    Histórico de conversa: {messages}
    """)

    conversation_model = wrap_model(m, conversation_prompt.format(messages=state["messages"]))
    response = await conversation_model.ainvoke(state, config)
    response_message = response.content.strip()

    state["messages"].append(AIMessage(response_message))
    return state

async def confirm_order(state: AgentState, config: RunnableConfig) -> AgentState:
    """Confirma os detalhes do pedido com o usuário e valida se estão completos."""
    logging.info("CONFIRM ORDER")
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))

    # Verifica se os detalhes do pedido estão presentes
    if "order_details" not in state:
        logging.error("Detalhes do pedido não estão definidos no estado.")
        raise ValueError("Detalhes do pedido não estão definidos no estado.")

    # Obtém os detalhes do pedido
    order_details = state["order_details"]

    # Pergunta ao usuário se ele deseja confirmar ou alterar o pedido
    confirm_prompt = SystemMessagePromptTemplate.from_template("""
    Aqui estão os detalhes do seu pedido:
    {order_details}

    Por favor, confirme se está tudo certo ou se deseja fazer alguma alteração.
    """)

    confirm_model = wrap_model(m, confirm_prompt.format(order_details=order_details))
    confirm_response = await confirm_model.ainvoke(state, config)
    confirm_message = confirm_response.content.strip()

    state["messages"].append(AIMessage(confirm_message))

    # Interpreta a intenção do usuário
    intent_prompt = SystemMessagePromptTemplate.from_template("""
    Analise a resposta do usuário e determine se ele deseja confirmar ou alterar o pedido, retorne:
    - "confirm" para confirmar o pedido.
    - "change" para alterar o pedido.
    Resposta do usuário: {user_response}
    """)

    intent_model = wrap_model(m, intent_prompt.format(user_response=confirm_message))
    intent_response = await intent_model.ainvoke(state, config)
    intent = intent_response.content.strip().lower()

    logging.info(f"Intenção detectada: {intent}")

    # Atualiza o estado com base na intenção detectada
    if intent == "confirm":
        state["confirm_order_state"] = "confirmed"
    elif intent == "change":
        state["confirm_order_state"] = "redirect_to_capture"
    else:
        # Caso a intenção não seja clara, mantém o estado atual
        pass

    return state

async def register_order(state: AgentState, config: RunnableConfig) -> AgentState:
    """Registra o pedido e finaliza o fluxo."""
    logging.info("REGISTER ORDER")

    final_message = (
        f"✅ Pedido registrado com sucesso!\n"
        "Agradecemos sua preferência! Seu pedido já está a caminho. 🚚\n"
        "Se quiser fazer outro pedido, é só mandar uma mensagem aqui no chat!"
    )

    logging.info("Adicionando mensagem final")
    state["messages"].append(AIMessage(
        content=final_message,
        response_metadata={
            "order_details": {
                "description": state.get("order_details"),
            }
        }
    ))

    return state

def check_confirmation(state: AgentState) -> str:
    """Verifica o estado de confirmação e decide o próximo nó."""
    if state["confirm_order_state"] == "confirmed":
        return "register_order"  # Redireciona para o nó de registro do pedido
    elif state["confirm_order_state"] == "redirect_to_capture":
        return "capture_order"   # Redireciona para o nó de captura do pedido
    else:
        return "confirm_order"   # Mantém no nó de confirmação

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