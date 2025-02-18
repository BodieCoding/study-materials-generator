from sklearn.feature_extraction.text import TfidfVectorizer
import asyncio
import ollama
import logging
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

class ChatSession:
    def __init__(self, materials, ollama_model):
        self.materials = materials
        self.ollama_model = ollama_model
        self.vector_embeddings = self._generate_embeddings(materials)
        logging.info("Chat session initialized with model: %s", self.ollama_model)
        logging.info("Size of embeddings: %s", self.vector_embeddings.shape)        

    def _generate_embeddings(self, materials):
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(materials)
        logging.info("Generated embeddings for materials")
        return embeddings

    async def ask_question(self, question):
        message = {'role': 'user', 'content': question}
        client = ollama.AsyncClient()
        response = await client.chat(model=self.ollama_model, messages=[message], stream=True)
        answer = ""
        async for part in response:
            answer += part['message']['content']
        logging.info("Answer: %s", answer)
        return answer

    async def start_chat(self):
        print("Welcome to the interactive Q&A session!")
        while True:
            question = input("Ask a question about your study materials (or type 'exit' to quit): ")
            if question.lower() == 'exit':
                print("Ending the chat session.")
                break
            answer = await self.ask_question(question)
            print(f"Answer: {answer}")