import os
from datasets import load_dataset

# Configurações do projeto
DATASET_NAME = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.pt" # Versão estática em português
OUTPUT_FILE = "data/raw/wikipedia_10gb.txt"
TARGET_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB exatos em bytes

def download_wikipedia():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print(f"Iniciando download em modo streaming da base {DATASET_NAME}...")
    
    # Conecta ao Hugging Face sem carregar tudo na memória
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=True)
    
    bytes_written = 0
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, article in enumerate(dataset):
            text = article.get("text", "")
            
            # Salva o texto limpo com quebra dupla para delimitar os documentos
            f.write(text + "\n\n")
            bytes_written += len(text.encode('utf-8'))
            
            # Feedback visual a cada 1000 artigos
            if i % 1000 == 0:
                print(f"Baixados: {bytes_written / (1024**3):.3f} GB / 10.000 GB")
                
            if bytes_written >= TARGET_BYTES:
                print("\nMeta de 10 GB de texto puro atingida com sucesso!")
                break

if __name__ == "__main__":
    download_wikipedia()