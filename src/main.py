import os
import streamlit as st
from search_engine import RAGSearchEngine, EMBEDDINGS_FILE
from llm_generator import RAGLLM


@st.cache_resource(show_spinner="Carregando indice...")
def load_resources(embeddings_mtime: float):
    #embeddings_mtime é usado coomo chave do cache

    engine = RAGSearchEngine()
    engine.load_data_and_build_index()

    return RAGLLM(engine)


def build_resources():
     # inicializa historico
    if "messages" not in st.session_state: 
        st.session_state.messages = []
    
    # registra historico
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

def awnser_stream(llm: RAGLLM, pergunta: str, chunks: list[dict]):
    for chunk in llm.generate_awnser(pergunta, chunks):
        content = chunk.message.content
        if content:
            yield content


def main():
    try:
        llm = load_resources(os.path.getmtime(EMBEDDINGS_FILE))
    except FileNotFoundError:
        st.error("Arquivos de dados não encontrados. Execute o embedder.py primeiro!")
        st.stop()

    build_resources()

    if prompt := st.chat_input("Como posso ajuda-lo hoje ?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.status("Pensando...", expanded=True) as status:
                contexto_chunks = llm.fetch_context(prompt)
                st.write("Trechos Encontrandos. Gerando resposta...")
                
            resposta = st.write_stream(awnser_stream(llm, prompt, contexto_chunks))
            status.update(label="Resposta gerada",  state="complete", expanded=False)

        st.session_state.messages.append({"role": "assistant", "content": resposta})

main()