import logging
from typing import Literal, cast

from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from core import get_model, llm, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AgentState(MessagesState, total=False):
    """Estado do agente para o FastFood."""
    order_details: str
    user_completion_feedback: str | None
    capture_order_state: str = "more_details"
    confirm_order_state: str = "change_order"

class CaptureOrderExtraction(BaseModel):
    step: Literal["order_ready", "more_details"] = Field(
        ...,
        description="""
        Estado de preparação do pedido:
        - 'order_ready': O pedido contém todos os detalhes necessários e está pronto para produção
        - 'more_details': Faltam informações ou há inconsistências que impedem a produção
        (campo obrigatório)
        """
    )
    reasoning: str = Field(
        ...,
        description="""
        Análise detalhada da situação do pedido:
        - Se 'order_ready': Confirmação dos detalhes completos do pedido
        - Se 'more_details': Lista clara dos itens faltantes ou inconsistências identificadas
        Deve fornecer orientações específicas sobre o que precisa ser ajustado quando aplicável.
        (campo obrigatório)
        """
    )


class ConfirmOrderExtraction(BaseModel):
    step: Literal["confirm_order", "change_order"] = Field(
        ...,
        description="""
        Decisão final do cliente:
        - 'confirm_order': Cliente aprovou todos os detalhes e quer prosseguir com o pedido
        - 'change_order': Cliente solicitou modificações no pedido atual
        (campo obrigatório)
        """
    )
    reasoning: str = Field(
        ...,
        description="""
        Contexto da decisão do cliente:
        - Para 'confirm_order': Pode incluir confirmação explícita ou aceitação tácita
        - Para 'change_order': Deve detalhar especificamente quais alterações foram solicitadas
        Incluir citações relevantes do cliente quando possível.
        (campo obrigatório)
        """
    )


background_prompt = SystemMessagePromptTemplate.from_template("""
    Você é o assistente digital da Lanchonete FASTFOOD. 

    INÍCIO DA CONVERSA:
    "Olá! Vamos montar seu pedido passo a passo. Você precisa informar:
    1. O que deseja pedir
    2. Quantidades
    3. Detalhes importantes (tamanho, modificações)

    EXEMPLO RÁPIDO:
    'Quero 1 X-Salada médio sem cebola e 2 refrigerantes pequenos'

    Podemos começar? O que vai pedir hoje?"
""")

capture_order_extraction_prompt = SystemMessagePromptTemplate.from_template("""
    Você é um assistente especializado em capturar pedidos de lanchonete com precisão. 

    REGRAS PARA PEDIDO COMPLETO:
    Um pedido é completo quando contém PELO MENOS 2 dos 9 elementos abaixo:
    ✓ Itens solicitados (obrigatório)
    ✓ Quantidades de cada item - opicional
    ✓ Tamanhos (quando aplicável) - opicional
    ✓ Modificações/preferências (ex: sem cebola)  - opicional
    ✓ Sabor (quando aplicável) - opicional
    ✓ Acompanhamentos (molhos extra, bordas, etc.) - opicional
    ✓ Adicionais (bacon extra, queijo extra, etc.) - opicional
                                                                            
    Um pedido completo não pode conter palavrões ou items não comestíveis.  
""")

confirm_order_extraction_prompt = SystemMessagePromptTemplate.from_template("""
    Você é um atendente de lanchonete responsável por confirmar com o cliente o pedido.
                                                                            
    Regras para confirmar pedidos:
    O Cliente pode confirmar o pedido ou solicitar alterações nele.
""")

def wrap_model(model: BaseChatModel, system_prompt: SystemMessage) -> RunnableSerializable[AgentState, AIMessage]:

    """Encapsula o modelo com um prompt de sistema."""
    preprocessor = RunnableLambda(
        lambda state: [system_prompt] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


async def background(state: AgentState, config: RunnableConfig) -> AgentState:
    """This node is to demonstrate the work done before a request is captured."""

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m, background_prompt.format())
    response = await model_runnable.ainvoke(state, config)

    return {"messages": [AIMessage(content=response.content)]}

async def capture_order(state: AgentState, config: RunnableConfig) -> AgentState:
    """Este nó captura o pedido do cliente e verifica se ele tem detalhes suficientes para ir para produção ou se alterações precisam ser feitas."""
    logging.info("CAPTURE ORDER NODE")

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))

    logging.info("Preparando modelo runnable com saída estruturada")
    model_runnable = wrap_model(
        m.with_structured_output(CaptureOrderExtraction), capture_order_extraction_prompt.format()
    )

    logging.info("Invocando modelo para extração de pedido")
    response = await model_runnable.ainvoke(state, config)
    response = cast(CaptureOrderExtraction, response)
    logging.info(f"Resposta do modelo recebida: {response}")

    if response.step == "more_details":
        logging.warning("Pedido precisa de mais detalhes - step is more_details")
        logging.info(f"Motivo fornecido pelo modelo: {response.reasoning}")
        capture_intention = interrupt(f"{response.reasoning}\n" "Quais alterações você deseja realizar no pedido?")
        state["messages"].append(HumanMessage(capture_intention))
        return await capture_order(state, config)

    return {
        "capture_order_state": response.step,
        "order_details": response.reasoning
    }

async def confirm_order(state: AgentState, config: RunnableConfig) -> AgentState:
    """This node examines the conversation history to determine if the customer wants to finalize the order or make further changes before sending it for delivery."""
    logging.info("CONFIRM ORDER NODE")

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(
        m.with_structured_output(ConfirmOrderExtraction), confirm_order_extraction_prompt.format()
    )
    response = await model_runnable.ainvoke(state, config)
    response = cast(ConfirmOrderExtraction, response)

    if response.step == "change_order":
        logging.warning("Usuário deseja alterar o pedido - step is change_order")
        logging.info(f"Motivo fornecido pelo modelo: {response.reasoning}")
        order_intention = interrupt(f"Você deseja alterar ou finalizar seu pedido?")
        state["messages"].append(HumanMessage(order_intention))
        return await confirm_order(state, config)

    return {
        "confirm_order_state": response.step
    }


async def register_order(state: AgentState, config: RunnableConfig) -> AgentState:
    """Registra o pedido e finaliza o fluxo."""
    logging.info("REGISTER ORDER")
    logging.info(f"Detalhes do pedido = {state.get("order_details")}")

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

def capture_route_confirmation(state: AgentState) -> str:
    if state["capture_order_state"] == "more_details":
        return "capture_order"
    elif state["capture_order_state"] == "order_ready":
        return "confirm_order" 

def confirm_route_confirmation(state: AgentState) -> str:
    if state["confirm_order_state"] == "change_order":
        return "capture_order"
    elif state["confirm_order_state"] == "confirm_order":
        return "register_order" 

# Cria o grafo do agente
agent = StateGraph(AgentState)
agent.add_node("background", background)
agent.add_node("capture_order", capture_order)
agent.add_node("confirm_order", confirm_order)
agent.add_node("register_order", register_order)

# Define o ponto de entrada
agent.set_entry_point("background")
agent.add_edge("background", "capture_order")

# Adiciona arestas condicionais
agent.add_conditional_edges(
    "capture_order",
    capture_route_confirmation,
    {
        "confirm_order": "confirm_order",
        "capture_order": "capture_order",
    },
)

agent.add_conditional_edges(
    "confirm_order",
    confirm_route_confirmation,
    {
        "register_order": "register_order",
        "capture_order": "capture_order",
    },
)

agent.add_edge("capture_order", "confirm_order")

agent.add_edge("register_order", END)

# Compila o agente
fastfood_agent = agent.compile(checkpointer=MemorySaver())
fastfood_agent.name = "fastfood-agent"