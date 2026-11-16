# https://docs.ollama.com/modelfile


from ollama import chat
from search_engine import RAGSearchEngine
import time

class RAGLLM:

    def __init__(self, search_engine: RAGSearchEngine):
        self.context = []
        self.sarch_engine = search_engine


    def _prompt(self, context: str, question: str):
        return f"""Responda à pergunta da meneira mais completa possível usando APENAS as informações do CONTEXTO abaixo. 
                Se o contexto não contiver informações suficientes para responder, diga exatamente: "Não encontrei informações na base de dados para responder a esta pergunta."
        
                CONTEXTO:
                {context}
        
                PERGUNTA:
                {question}
                """


    def fetch_context(self, user_question: str, k: int = 5):
        return self.sarch_engine.search(user_question, k)


    def generate_awnser(self, user_question: str):
        context = self.fetch_context(user_question)
        prompt = self._prompt(context, user_question)

        start_time = time.perf_counter()
        first_chunk = None
        last_chunk = None

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
        for chunk in answer_stream:
            if first_chunk is None:
                first_chunk = time.perf_counter() - start_time
            print(chunk.message.content, end="", flush=True)

            if chunk.done:
                last_chunk = chunk

        total = time.perf_counter() - start_time
        ns = 1e9

        return {
            "thinking_time": start_time,
            "loadmodel_time": (last_chunk.load_duration / ns),
            "time2process": (last_chunk.prompt_eval_duration / ns),
        }