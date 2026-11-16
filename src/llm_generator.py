# https://docs.ollama.com/modelfile


from ollama import chat
from search_engine import RAGSearchEngine

class RAGLLM:

    def __init__(self, search_engine: RAGSearchEngine):
        self.context = []
        self.search_engine = search_engine


    def _prompt(self, context: str, question: str):
        return f"""Responda à pergunta da meneira mais completa possível usando APENAS as informações do CONTEXTO abaixo. 
                Se o contexto não contiver informações suficientes para responder, diga exatamente: "Não encontrei informações na base de dados para responder a esta pergunta."
        
                CONTEXTO:
                {context}
        
                PERGUNTA:
                {question}
                """


    def fetch_context(self, user_question: str, k: int = 5):
        return self.search_engine.search(user_question, k)


    def generate_awnser(self, user_question: str):
        context = self.fetch_context(user_question)
        prompt = self._prompt(context, user_question)


        answer = chat(
            model="qwen3.5:2b",
            messages=[{"role": "user", "content": prompt}],
            think=False,
            options={
                "temperature": 0.1,
                "top_p": 0.9,
            }
        )
        return answer.message.content