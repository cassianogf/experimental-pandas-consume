from __future__ import annotations

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """\
Você é um analista de dados especialista em Pandas.
Recebe a descrição de um DataFrame chamado `df` e uma pergunta do usuário.

Seu trabalho é gerar **apenas** código Python/Pandas que responda à pergunta.

Regras:
- O DataFrame já está carregado na variável `df`. NÃO tente ler arquivos.
- A última expressão do código DEVE ser atribuída a uma variável chamada `resultado`.
  `resultado` pode ser um valor escalar, uma Series ou um DataFrame.
- Use apenas pandas e a stdlib do Python — nenhuma outra lib.
- Se a pergunta pedir um gráfico ou visualização, gere um DataFrame resumido
  que possa ser exibido como tabela ou gráfico de barras pelo Streamlit.
- Responda SOMENTE com o bloco de código, sem explicação, sem markdown fences.
- Valores monetários na coluna `valor_pedido` já são float.
- Colunas booleanas (`pix`, `clube_yooga`, `comprou_cupom`) contêm "SIM"/"NÃO".
- Quando o usuário mencionar um nome (de restaurante, cliente, etc), NUNCA use
  igualdade exata (==). Use `str.contains(termo, case=False, na=False)` para
  busca parcial, pois o usuário pode abreviar ou omitir partes do nome.
"""


def _build_schema_description(df) -> str:
    lines = [f"Colunas ({len(df.columns)}):"]
    for col in df.columns:
        dtype = df[col].dtype
        sample = df[col].dropna().head(3).tolist()
        lines.append(f"  - {col} ({dtype}): ex {sample}")
    lines.append(f"\nTotal de linhas: {len(df)}")
    return "\n".join(lines)


def summarize_result(result) -> str:
    """Gera um resumo textual curto do resultado para usar como contexto."""
    import pandas as pd

    if isinstance(result, pd.DataFrame):
        preview = result.head(5).to_string(index=False)
        return f"DataFrame ({result.shape[0]} linhas x {result.shape[1]} colunas):\n{preview}"
    if isinstance(result, pd.Series):
        preview = result.head(5).to_string()
        return f"Series ({len(result)} itens):\n{preview}"
    return str(result)


def ask_llm(
    question: str,
    df,
    *,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    history: list[dict] | None = None,
) -> str:
    """Envia a pergunta + schema do df para a LLM e retorna o código Pandas gerado."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    schema = _build_schema_description(df)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    schema_msg = f"### Schema do DataFrame `df`\n{schema}"

    if history:
        messages.append({"role": "user", "content": schema_msg})
        messages.append({"role": "assistant", "content": "Entendido. Pode perguntar."})
        messages.extend(history)
        messages.append({"role": "user", "content": question})
    else:
        messages.append({"role": "user", "content": f"{schema_msg}\n\n### Pergunta\n{question}"})

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )

    code = response.choices[0].message.content.strip()
    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:])
    if code.endswith("```"):
        code = "\n".join(code.split("\n")[:-1])
    return code
