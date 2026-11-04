import os
import time
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Configurações de Arquivos
INPUT_FILE = "data/raw/corpus_10gb.txt"
OUTPUT_EMBEDDINGS = "data/processed/embeddings.npy"
OUTPUT_CHUNKS = "data/processed/chunks.jsonl"

# Parâmetros de Quebra (Chunking) e Performance
CHUNK_SIZE = 1000  # Caracteres por bloco
OVERLAP = 200      # Sobreposição de contexto
BATCH_SIZE = 64    # Lote para a GPU processar de uma vez
MAX_CHUNKS = 50000 # Trava de segurança: processar ~50 mil blocos para o experimento

def get_chunks(filepath, chunk_size, overlap, max_chunks):
    """Lê o arquivo de 10GB sequencialmente e gera blocos de texto na memória."""
    buffer = ""
    chunks_yielded = 0
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            buffer += line + " "
            
            while len(buffer) >= chunk_size:
                if chunks_yielded >= max_chunks:
                    return
                
                yield buffer[:chunk_size]
                buffer = buffer[chunk_size - overlap:]
                chunks_yielded += 1

def run_embedder():
    os.makedirs(os.path.dirname(OUTPUT_EMBEDDINGS), exist_ok=True)
    
    print("Carregando o modelo de linguagem na GPU...")
    # Modelo multilíngue rápido e eficiente para Português
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    print(f"Iniciando vetorização (Limite configurado: {MAX_CHUNKS} chunks)...")
    start_time = time.time()
    
    chunk_generator = get_chunks(INPUT_FILE, CHUNK_SIZE, OVERLAP, MAX_CHUNKS)
    all_embeddings = []
    batch = []
    total_processed = 0
    
    with open(OUTPUT_CHUNKS, "w", encoding="utf-8") as f_out:
        for chunk in chunk_generator:
            batch.append(chunk)
            
            # Salva o texto associado a um ID para o RAG recuperar depois
            json.dump({"id": total_processed + len(batch) - 1, "text": chunk}, f_out, ensure_ascii=False)
            f_out.write("\n")
            
            # Quando o lote enche, envia para a GPU
            if len(batch) == BATCH_SIZE:
                emb = model.encode(batch, convert_to_numpy=True)
                all_embeddings.append(emb)
                total_processed += len(batch)
                batch = []
                
                if total_processed % 5000 == 0:
                    print(f"Vetorizados: {total_processed} / {MAX_CHUNKS}...")
        
        # Processa o restante se o último lote ficar incompleto
        if batch:
            emb = model.encode(batch, convert_to_numpy=True)
            all_embeddings.append(emb)
            total_processed += len(batch)
            
    # Empilha todos os lotes e salva a matriz matemática (.npy) no disco
    print("\nSalvando a matriz vetorial no disco...")
    final_matrix = np.vstack(all_embeddings)
    np.save(OUTPUT_EMBEDDINGS, final_matrix)
    
    elapsed = time.time() - start_time
    print(f"Concluído! {total_processed} chunks processados em {elapsed:.2f} segundos.")
    print(f"Formato da matriz matemática gerada: {final_matrix.shape}")

if __name__ == "__main__":
    run_embedder()