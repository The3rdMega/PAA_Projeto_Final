import streamlit as st
from search_engine import RAGSearchEngine
from llm_generator import RAGLLM

def search(llm: RAGLLM):
    if "input_area" not in st.session_state:
        return
    pergunta = st.session_state["input_area"]
    resposta = ""
    if pergunta:
        resposta = llm.generate_awnser(pergunta)
        st.session_state["output_area"] = resposta


def load_resources():
    # limpa cache em caso de embedder.py for rodado
    st.cache_resource.clear()

    engine = RAGSearchEngine()
    llm = RAGLLM(engine)
    engine.load_data_and_build_index()
    return llm



try:
    llm = load_resources()
except FileNotFoundError:
    st.error("Arquivos de dados não encontrados. Execute o embedder.py primeiro!")
    st.stop()



entrada = st.text_input(
    label="Area de perguntas",
    value="",
    type="search",
    key="input_area",
    on_change= search(llm)
    )

st.text_area(
    label="Output Log",
    value=None,
    height=1000, 
    disabled=True,
    key="output_area"
    )

#st.button(
#    label= "Pesquisar",
#    on_click= search(engine)
#)