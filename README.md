# 🧰 FastFood Agent API

Uma ferramenta completa para rodar um serviço de agente de IA baseado no LangGraph, FastAPI e Streamlit.

Este projeto fornece uma estrutura robusta para criar e executar agentes baseados em LangGraph. Um dos principais componentes é o **agent para cadastro de pedidos**, que permite aos usuários registrar pedidos de forma automatizada através de uma API FastAPI, garantindo escalabilidade e integração eficiente com diferentes sistemas.

## 📌 Visão Geral

### Principais Componentes
- **Agente IA com LangGraph**: Um agente personalizável para lidar com interações e automação.
- **API FastAPI para cadastro de pedidos**: Permite registrar pedidos dos usuários de forma rápida e segura.
- **Cliente Python**: Facilita a comunicação com a API.
- **Interface em Streamlit**: Uma interface amigável para interações com o agente.

### 🔧 Estrutura do Projeto

```plaintext
📂 agent-service-toolkit
├── 📂 src
│   ├── 📂 agents          # Implementação dos agentes
│   ├── 📂 schema          # Definição dos esquemas de dados
│   ├── 📂 core            # Módulos centrais como configurações e LLM
│   ├── 📂 service         # API FastAPI, incluindo o agent de pedidos
│   │   ├── service.py     # Serviço principal FastAPI
│   │   ├── orders.py      # API para cadastro de pedidos
│   ├── 📂 client          # Cliente Python para consumir a API
│   ├── streamlit_app.py   # Interface de chat em Streamlit
├── 📂 tests               # Testes automatizados
└── README.md
```

## 🚀 Como Rodar o Projeto

### 1️⃣ Clonar o Repositório
```sh
git clone https://github.com/ThiagoDias/agent-service-toolkit.git
cd agent-service-toolkit
```

### 2️⃣ Configurar as Variáveis de Ambiente
Crie um arquivo `.env` e adicione suas credenciais:
```sh
echo 'OPENAI_API_KEY=sua_chave_openai' >> .env
```

### 3️⃣ Rodar com Python
```sh
pip install uv
uv sync --frozen
source .venv/bin/activate
python src/service/service.py
```

### 4️⃣ Rodar a Interface Streamlit
```sh
streamlit run src/streamlit_app.py
```

### 5️⃣ Testar a API de Cadastro de Pedidos
```sh
curl -X POST "http://127.0.0.1:8000/orders" -H "Content-Type: application/json" -d '{"item": "Produto X", "quantidade": 2, "usuario": "Thiago"}'
```
