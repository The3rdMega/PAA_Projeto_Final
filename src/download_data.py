import os
from datasets import load_dataset

# Configurações
OUTPUT_FILE = "data/raw/corpus_10gb.txt" # Nome atualizado refletindo a união de bases
TARGET_BYTES = 10 * 1024 * 1024 * 1024   # 10 GB

def extract_and_append(dataset_path, dataset_config, split, current_bytes):
    """Função modular para extrair texto de qualquer base do Hugging Face"""
    print(f"\nIniciando extração da base: {dataset_path} ({dataset_config})...")
    dataset = load_dataset(dataset_path, dataset_config, split=split, streaming=True)
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i, article in enumerate(dataset):
            text = article.get("text", "")
            if not text:
                continue
            
            f.write(text + "\n\n")
            current_bytes += len(text.encode('utf-8'))
            
            if i % 1000 == 0:
                print(f"Progresso: {current_bytes / (1024**3):.3f} GB / 10.000 GB")
                
            if current_bytes >= TARGET_BYTES:
                return current_bytes
                
    print(f"Fim da base {dataset_path} atingido.")
    return current_bytes

def build_corpus():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 1. Verifica o que já temos no disco
    current_bytes = os.path.getsize(OUTPUT_FILE) if os.path.exists(OUTPUT_FILE) else 0
    print(f"Tamanho inicial no disco: {current_bytes / (1024**3):.3f} GB")

    if current_bytes >= TARGET_BYTES:
        print("A meta de 10 GB já foi atingida!")
        return

    # 2. FASE 1: Wikipedia em Português
    # O script só tenta baixar a Wiki do zero se o arquivo estiver completamente vazio.
    # Isso evita duplicar os dados da Wiki caso o script seja interrompido e reiniciado.
    if current_bytes == 0:
        current_bytes = extract_and_append("wikimedia/wikipedia", "20231101.pt", "train", current_bytes)
    else:
        print("O arquivo já contém dados. Pulando a extração inicial da Wikipedia para evitar duplicação.")

    # 3. FASE 2: Complemento com mC4 em Português
    if current_bytes < TARGET_BYTES:
        faltam_gb = (TARGET_BYTES - current_bytes) / (1024**3)
        print(f"\nFaltam {faltam_gb:.3f} GB. Alternando para o modo de complemento...")
        current_bytes = extract_and_append("allenai/c4", "pt", "train", current_bytes)
        
    # Conclusão
    if current_bytes >= TARGET_BYTES:
        print("\nMeta de 10 GB de texto puro atingida com sucesso!")

if __name__ == "__main__":
    build_corpus()