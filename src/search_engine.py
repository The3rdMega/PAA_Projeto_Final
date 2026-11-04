import time
import json
import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer

# Caminhos dos arquivos processados
EMBEDDINGS_FILE = "data/processed/embeddings.npy"
CHUNKS_FILE = "data/processed/chunks.jsonl"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIMENSIONALITY = 384  # Número de dimensões do modelo MiniLM

class RAGSearchEngine:
    def __init__(self):
        print("Carregando modelo de linguagem...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.chunks_data = []

    def load_data_and_build_index(self):
        print(f"\nCarregando matriz vetorial de {EMBEDDINGS_FILE}...")
        start_time = time.time()
        
        # 1. Carrega os vetores matemáticos
        embeddings = np.load(EMBEDDINGS_FILE)
        num_elements = embeddings.shape[0]
        
        # 2. Carrega os textos correspondentes para a memória
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                self.chunks_data.append(json.loads(line)["text"])
                
        print(f"Dados carregados. Construindo grafo HNSW para {num_elements} vetores...")
        
        # 3. Inicializa o algoritmo HNSW
        # space='cosine' mede a similaridade angular entre os vetores (padrão para textos)
        self.index = hnswlib.Index(space='cosine', dim=DIMENSIONALITY)
        
        # Parâmetros vitais para PAA (Projeto e Análise de Algoritmos):
        # M: Quantidade de arestas (conexões) por nó no grafo. (Maior = mais preciso, porém mais RAM).
        # ef_construction: Tamanho da lista dinâmica durante a construção do grafo. (Maior = melhor qualidade, construção mais lenta).
        self.index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        
        # Insere os dados no grafo
        self.index.add_items(embeddings)
        
        # ef: Tamanho da fila de prioridade na hora da busca. (Controla o trade-off entre tempo de busca e precisão).
        self.index.set_ef(50)
        
        elapsed = time.time() - start_time
        print(f"Grafo construído com sucesso em {elapsed:.2f} segundos!")

    def search(self, query, k=3):
        """Transforma a pergunta em vetor e busca os 'k' nós mais próximos no grafo."""
        print(f"\nBuscando por: '{query}'")
        start_time = time.time()
        
        # Vetoriza a pergunta
        query_vector = self.model.encode([query], convert_to_numpy=True)
        
        # Realiza a busca KNN (K-Nearest Neighbors) no grafo HNSW
        labels, distances = self.index.knn_query(query_vector, k=k)
        
        elapsed = time.time() - start_time
        print(f"Busca concluída em {elapsed:.4f} segundos.")
        
        results = []
        for i in range(k):
            idx = labels[0][i]
            # A distância de cosseno varia de 0 (idêntico) a 2 (oposto). Similaridade é 1 - distância.
            similarity = 1 - distances[0][i] 
            results.append({
                "texto": self.chunks_data[idx],
                "similaridade": round(similarity * 100, 2)
            })
            
        return results

if __name__ == "__main__":
    # Teste prático do motor de busca
    engine = RAGSearchEngine()
    
    try:
        engine.load_data_and_build_index()
    except FileNotFoundError:
        print("\nErro: Arquivos de dados não encontrados. Execute o embedder.py primeiro!")
        exit(1)
        
    # Laço interativo para você testar perguntas
    print("\nMotor de busca pronto! Digite 'sair' para encerrar.")
    #TODO: Depois substituir essa parte por um frontend completo, provavelmente usando Flask
    while True:
        pergunta = input("\nFaça uma pergunta: ")
        if pergunta.lower().strip() == 'sair':
            break
            
        resultados = engine.search(pergunta, k=3)
        
        print("\n--- TOP 3 RESULTADOS ENCONTRADOS ---")
        for i, res in enumerate(resultados, 1):
            print(f"\n{i}. [Similaridade: {res['similaridade']}%]")
            print(res['texto'])