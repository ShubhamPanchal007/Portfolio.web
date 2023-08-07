from langchain.prompts import PromptTemplate
import queue
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

prompt_template = """You are an AI Avatar of Shubham Panchal. So answer every question in the first person narrative as shubham panchal. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer the question as Shubham Panchal in the first person narrative"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)



# Thread Generator Class

class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        # removes the element from the queue and returns it.
        item = self.queue.get()
        if item is StopIteration:
            raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)

# ChainStreamHandler class

class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs):
        self.gen.send(token)

