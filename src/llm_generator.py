# https://docs.ollama.com/modelfile


from ollama import chat
from search_engine import RAGSearchEngine

class RAGLLM:

    def __init__(self, search_engine: RAGSearchEngine):
        self.search_engine = search_engine


    def fetch_context(self, user_question: str, k: int = 5) -> list[dict]:
        """invoca metodo de search_engine que realiza busca no grafo HNSW usando similirdade de cosseno"""
        return self.search_engine.search(user_question, k)


    def _format_context(self, chunks: list[dict]):
        return "\n\n".join(
            f"[Trecho: {i}]\n{c['texto']}" for i, c in enumerate(chunks, start=1)
        )

    def _prompt(self, context: str, question: str):
        return (
            "Responda à pergunta da maneira mais completa possível usando APENAS "
            "as informações do CONTEXTO abaixo.\n"
            "Se o contexto não contiver informações suficientes, diga exatamente: "
            '"Não encontrei informações na base de dados para responder a esta pergunta."\n\n'
            f"CONTEXTO:\n{context}\n\n"
            f"PERGUNTA:\n{question}\n"
        )


    def generate_awnser(self, user_question: str, chunks: list[dict]):
        context = self._format_context(chunks)
        prompt = self._prompt(context, user_question)


        answer_stream = chat(
            model="qwen3.5:2b",
            messages=[{"role": "user", "content": prompt}],
            think=False,
            stream=True,
            options={
                "temperature": 0.1,
                "top_p": 0.9,
            }
        )
        return answer_stream