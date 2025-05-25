import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama
from streamlit_logger import get_logger

class ChatSession:
    def __init__(self, materials, ollama_model):
        self.materials = materials
        self.ollama_model = ollama_model
        self.logger = get_logger(f"streamlit_logger.{__name__}")
        self.vectorizer = TfidfVectorizer()
        self.vector_embeddings = self._generate_embeddings(materials)
        self.logger.info("Chat session initialized with model: %s", self.ollama_model)
        self.logger.info("Size of embeddings: %s", self.vector_embeddings.shape)

    def _generate_embeddings(self, materials):
        embeddings = self.vectorizer.fit_transform(materials)
        self.logger.info("Generated embeddings for materials")
        return embeddings

    def _find_most_relevant_material(self, question):
        question_embedding = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_embedding, self.vector_embeddings)
        most_relevant_index = similarities.argmax()
        self.logger.info("Most relevant material index: %s", most_relevant_index)
        return self.materials[most_relevant_index]

    async def ask_question(self, question):
        try:
            relevant_material = self._find_most_relevant_material(question)
            message = {'role': 'user', 'content': f"{question}\n\nContext: {relevant_material}"}
            client = ollama.AsyncClient()
            response = await client.chat(model=self.ollama_model, messages=[message], stream=True)
            answer = ""
            async for part in response:
                if 'role' in part and part['role'] == 'assistant':
                    answer += part['content']
            return answer
        except ollama.ResponseError as e:
            self.logger.error("Error in response: %s", e)
            st.error(f"Error in response: {e}")
            return "Error in response."

    async def start_chat(self):
        st.markdown(
            "<h2 style='text-align: center; color: #4CAF50; font-family: Arial;'>Hermione🪶</h2>",
            unsafe_allow_html=True,
        )
        
        # Initialize message history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi! How may I help you with your study materials?"}
            ]

        # Display the chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle user input
        if user_input := st.chat_input("Ask a question about your study materials (or type 'exit' to quit):"):
            # Add user message to session state
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            
            if user_input.lower() == 'exit':
                st.write("Ending the chat session.")
                return
            
            # Generate assistant response
            answer = await self.ask_question(user_input)
            
            # Add assistant response to session state
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(answer)

# Example usage
# if __name__ == "__main__":
#     import asyncio
#     materials = [
#         "A short article on the principles of object-oriented programming (OOP).",
#         "Another text about machine learning basics."
#     ]
#     chat_session = ChatSession(materials, "orca-mini")
#     asyncio.run(chat_session.start_chat())