import search_engine, streamlit

def search(engine : search_engine.RAGSearchEngine):
    if "input_area" not in streamlit.session_state:
        return
    pergunta = streamlit.session_state["input_area"]
    if pergunta == "":
        streamlit.session_state["output_area"] = "deu nada"
    saida = "# TOP 3 RESULTADOS ENCONTRADOS\n"
    resultados = engine.search(str(pergunta), k=3)
    for i, res in enumerate(resultados, 1):
        saida += f"{i}. [Similaridade: {res['similaridade']}%]\n"
        saida += res['texto'] + "\n"
    streamlit.session_state["output_area"] = saida


engine = search_engine.RAGSearchEngine()
try:
    engine.load_data_and_build_index()
except FileNotFoundError:
    streamlit.write("\nErro: Arquivos de dados não encontrados. Execute o embedder.py primeiro!")
    exit(1)

entrada = streamlit.text_input(
    label="Area de perguntas",
    value="",
    type="search",
    key="input_area",
    on_change= search(engine)
    )

streamlit.text_area(
    label="Output Log",
    value=None,
    height=1000, 
    disabled=True,
    key="output_area"
    )




#streamlit.button(
#    label= "Pesquisar",
#    on_click= search(engine)
#)