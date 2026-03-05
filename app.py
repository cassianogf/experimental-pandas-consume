import streamlit as st
import pandas as pd
from llm import ask_llm, summarize_result

DEFAULT_MODEL = "gpt-4o"


def parse_dataframe(raw) -> pd.DataFrame:
    df = pd.read_csv(raw, sep="\t", encoding="utf-8")
    for col in df.columns:
        if df[col].dtype == "object":
            converted = df[col].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
            try:
                df[col] = converted.astype(float)
            except (ValueError, TypeError):
                pass
    return df


def execute_code(code: str, df: pd.DataFrame):
    """Executa o código gerado pela LLM num namespace isolado e retorna `resultado`."""
    namespace = {"df": df, "pd": pd}
    exec(code, namespace)
    return namespace.get("resultado", "Nenhuma variável `resultado` foi definida.")


def render_result(result):
    """Renderiza o resultado de forma inteligente dependendo do tipo."""
    if isinstance(result, pd.DataFrame):
        if len(result) <= 30 and len(result.columns) <= 3 and result.select_dtypes("number").shape[1] >= 1:
            st.bar_chart(result.set_index(result.columns[0]) if result.select_dtypes("number").shape[1] < len(result.columns) else result)
        st.dataframe(result, use_container_width=True)
    elif isinstance(result, pd.Series):
        st.dataframe(result.to_frame(), use_container_width=True)
    else:
        st.markdown(f"**Resultado:** `{result}`")


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Yooga · Análise de Consumo", page_icon="🍔", layout="wide")

st.title("Yooga · Análise de Consumo")
st.caption("Faça perguntas em português sobre os dados de consumo dos clientes.")

uploaded = st.file_uploader("Importe seu arquivo .txt (tabulado por tab)", type=["txt", "csv", "tsv"])

if uploaded is not None:
    file_id = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("_uploaded_id") != file_id:
        st.session_state["df"] = parse_dataframe(uploaded)
        st.session_state["messages"] = []
        st.session_state["_uploaded_id"] = file_id

if "df" not in st.session_state:
    st.info("Envie um arquivo para começar.")
    st.stop()

df = st.session_state["df"]

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Sobre os dados")
    st.metric("Linhas", f"{len(df):,}".replace(",", "."))

    id_cols = [c for c in df.columns if "id_cliente" in c.lower() or "cliente" in c.lower()]
    if id_cols:
        st.metric("Clientes únicos", f"{df[id_cols[0]].nunique():,}".replace(",", "."))

    rest_cols = [c for c in df.columns if "restaurante" in c.lower()]
    if rest_cols:
        id_col = next((c for c in rest_cols if "id" in c.lower()), rest_cols[0])
        st.metric("Restaurantes", f"{df[id_col].nunique():,}".replace(",", "."))

    with st.expander("Colunas disponíveis"):
        for col in df.columns:
            st.code(col, language=None)

    with st.expander("Amostra dos dados"):
        st.dataframe(df.head(10), use_container_width=True)

    st.divider()
    model = st.selectbox("Modelo LLM", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"], index=0)

    chat_mode = st.radio(
        "Modo do chat",
        ["Isolado", "Contexto"],
        index=0,
        help="Isolado: cada pergunta é independente. Contexto: a IA lembra das perguntas anteriores.",
    )
    prev_mode = st.session_state.get("chat_mode")
    st.session_state["chat_mode"] = chat_mode
    if prev_mode is not None and prev_mode != chat_mode:
        st.session_state["messages"] = []

    st.divider()
    st.caption("Dica: a IA gera código Pandas para responder. O cálculo é exato.")

# ── Sugestões ─────────────────────────────────────────────────────────────────

SUGGESTIONS = [
    "Quais clientes pediram em mais de um restaurante?",
    "Qual restaurante mais se repete entre os clientes?",
    "Qual o ticket médio por restaurante?",
    "Quais clientes usam PIX com mais frequência?",
    "Quais clientes fazem parte do Clube Yooga e em quais restaurantes eles pedem?",
]

# ── Chat state ────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("code"):
            with st.expander("Código gerado"):
                st.code(msg["code"], language="python")
        if msg.get("result") is not None:
            render_result(msg["result"])
        elif msg.get("error"):
            st.error(msg["error"])
        else:
            st.markdown(msg["content"])

# ── Sugestões como botões ─────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("#### Sugestões de perguntas")
    cols = st.columns(len(SUGGESTIONS))
    for i, suggestion in enumerate(SUGGESTIONS):
        if cols[i].button(suggestion, key=f"sug_{i}", use_container_width=True):
            st.session_state["pending_question"] = suggestion
            st.rerun()

# ── Input ─────────────────────────────────────────────────────────────────────

pending = st.session_state.pop("pending_question", None)
question = st.chat_input("Ex: Quais clientes pediram em mais de um restaurante?")
active_question = pending or question

if active_question:
    st.session_state.messages.append({"role": "user", "content": active_question})
    with st.chat_message("user"):
        st.markdown(active_question)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            history = None
            if chat_mode == "Contexto":
                history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        history.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "assistant" and msg.get("code"):
                        summary = summarize_result(msg["result"]) if msg.get("result") is not None else "Erro na execução"
                        history.append({"role": "assistant", "content": f"{msg['code']}\n\n# Resultado:\n{summary}"})

            try:
                code = ask_llm(active_question, df, model=model, history=history or None)
            except Exception as e:
                st.error(f"Erro ao chamar a LLM: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "", "error": str(e)})
                st.stop()

            with st.expander("Código gerado"):
                st.code(code, language="python")

            try:
                result = execute_code(code, df)
            except Exception as e:
                st.error(f"Erro ao executar o código: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "", "code": code, "error": str(e)})
                st.stop()

            render_result(result)
            st.session_state.messages.append({"role": "assistant", "content": "", "code": code, "result": result})
