from search_engine import RAGSearchEngine
from llm_generator import RAGLLM


if __name__ == "__main__":
    # Teste prático do motor de busca
    engine = RAGSearchEngine()
    llm = RAGLLM(engine)
    
    try:
        engine.load_data_and_build_index()
    except FileNotFoundError:
        print("\nErro: Arquivos de dados não encontrados. Execute o embedder.py primeiro!")
        exit(1)
        
    # Laço interativo para você testar perguntas
    print("\nMotor de busca pronto! Digite 'sair' para encerrar.")
    #TODO: Depois substituir essa parte por um frontend completo, provavelmente usando Flask


    while True:
        pergunta = input("\nVoce: ")
        if pergunta.lower().strip() == 'sair':
            break

        if not pergunta.strip():
            continue
            
        stats = llm.generate_awnser(pergunta)
        print(stats)
