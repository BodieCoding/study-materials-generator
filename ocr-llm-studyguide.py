import streamlit as st
import base64
import requests
from PIL import Image
import os
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import asyncio
import nest_asyncio
import ollama
import PyPDF2
from streamlit_logger import get_logger

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Get the custom logger
logger = get_logger()

class OCRProcessor:
    TEXT_EXTRACTION_PROMPT = """You are an advanced OCR tool. Your task is to accurately transcribe the text from the provided image.
    Please follow these guidelines to ensure the transcription is as correct as possible:
    1. **Preserve Line Structure**: Transcribe the text exactly as it appears in the image, including line breaks.
    2. **Avoid Splitting Words**: Ensure that words are fully formed, even if they appear across two lines or are partially obscured. Join word parts together appropriately to create complete words.
    3. **Correct Unnatural Spacing**: Do not add extra spaces between characters in a word. Make sure the words are spaced naturally, without any unwanted breaks or gaps.
    4. **Recognize and Correct Word Breaks**: If any word is mistakenly broken into parts, join them correctly to produce a natural, readable word. 
    5. **No Additional Comments or Analysis**: Provide only the raw text as it appears in the image, without any additional analysis, comments, or summaries.
    6. **Output as a Block of Text**: Output the entire transcribed text as a block, maintaining the line breaks, but ensuring that each word appears as it should, with correct spelling, no character-level splits, and no hyphenations unless they appear naturally in the image."""

    async def process_files_in_directory(self, directory):
        extracted_text = []
        tasks = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path in self.manifest:
                    continue
                if file.lower().endswith('.png'):
                    tasks.append(self.process_png(file_path))
                elif file.lower().endswith('.pdf'):
                    tasks.append(self.process_pdf(file_path))
                self.manifest[file_path] = True
        results = await asyncio.gather(*tasks)
        extracted_text.extend(results)
        self.save_manifest()
        logger.info("Processed files in directory: %s", directory)
        return extracted_text
    
    async def process_png(self, file_path):
        logger.info("Processing PNG file: %s", file_path)
        text = await self.extract_text_from_image(file_path)
        return text

    async def process_pdf(self, file_path):
        logger.info("Processing PDF file: %s", file_path)
        text = await self.extract_text_from_pdf(file_path)
        return text
    
    async def extract_text_from_pdf(self, file_path):
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfFileReader(file)
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extract_text()
        logger.info("Extracted text from PDF: %s", file_path)
        return text
    
    def encode_image_to_base64(self, image_path):
        """Convert an image file to a base64 encoded string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def perform_ocr(self, image_path):
        """Perform OCR on the given image using the provided API."""
        base64_image = self.encode_image_to_base64(image_path)

        # API request payload
        payload = {
            "model": "llama3.2-vision",
            "messages": [
                {
                    "role": "user",
                    "content": self.TEXT_EXTRACTION_PROMPT,
                    "images": [base64_image],
                },
            ],
        }

        # Send POST request to the API
        response = requests.post(
            "http://localhost:11434/api/chat",
            headers={"Content-Type": "application/json"},
            json=payload,
        )

        if response.status_code == 200:
            try:
                response_json = response.json()
                # Extract the content from the response
                choices = response_json.get("choices", [])
                if choices and "message" in choices[0] and "content" in choices[0]["message"]:
                    return choices[0]["message"]["content"].strip()
                else:
                    return "No text found in the image."
            except json.JSONDecodeError:
                st.error("Failed to parse JSON response.")
                st.write("Raw Response:", response.text)
                return None
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None

    def save_response(self, study_guide, image_name, response):
        """Save the OCR response to a text file in the study guide directory."""
        study_guide_dir = os.path.join("study_guides", study_guide)
        os.makedirs(study_guide_dir, exist_ok=True)
        response_file = os.path.join(study_guide_dir, f"{image_name}.txt")
        with open(response_file, "w") as f:
            f.write(response)

class MaterialGenerator:
    def __init__(self, output_file='materials.json'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        self.qg_tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-base-qg-hl")
        self.qg_model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-qg-hl")
        self.output_file = output_file

    def generate_materials(self, extracted_texts):
        summaries = self.generate_summary(extracted_texts)
        vocab_list = self.create_vocabulary_list(extracted_texts)
        practice_questions = self.generate_practice_questions(extracted_texts)
        materials = {
            'summaries': summaries,
            'vocab_list': vocab_list,
            'practice_questions': practice_questions
        }
        self.save_materials(materials)
        return materials

    def generate_summary(self, texts):
        summaries = []
        for text in texts:
            try:
                # Adjust max_length based on the input length
                input_length = len(text.split())
                max_length = min(130, input_length)
                summary = self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)
                if summary and len(summary) > 0:
                    logger.info("Generated summary for text")
                    summaries.append(summary[0]['summary_text'])
                else:
                    logger.warning("Summarizer returned an empty result")
                    summaries.append("Summary not available.")
            except Exception as e:
                logger.error("Error generating summary: %s", e)
                summaries.append("Summary not available.")
        return summaries

    def create_vocabulary_list(self, texts):
        words = ' '.join(texts).split()
        vocab_counter = Counter(words)
        vocab_list = [(word, count) for word, count in vocab_counter.most_common()]
        logger.info("Generated vocabulary list")
        return vocab_list

    def generate_practice_questions(self, texts):
        questions = []
        for text in texts:
            try:
                inputs = self.qg_tokenizer.encode("generate questions: " + text, return_tensors="pt", max_length=512, truncation=True)
                outputs = self.qg_model.generate(inputs, max_length=150, num_return_sequences=5, num_beams=5)
                for output in outputs:
                    question = self.qg_tokenizer.decode(output, skip_special_tokens=True)
                    questions.append(question)
                logger.info("Generated practice questions for text")
            except Exception as e:
                logger.error("Error generating practice questions: %s", e)
                questions.append("Could not generate questions.")
        return questions

    def save_materials(self, materials):
        with open(self.output_file, 'w') as file:
            json.dump(materials, file)
            logger.info("Materials saved to %s", self.output_file)

    def load_materials(self):
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r') as file:
                return json.load(file)
        return {}

    def format_materials(self, materials):
        formatted_output = "Summaries:\n"
        for summary in materials['summaries']:
            formatted_output += f"- {summary}\n"

        formatted_output += "\nVocabulary List:\n"
        for word, count in materials['vocab_list']:
            formatted_output += f"{word}: {count}\n"

        formatted_output += "\nPractice Questions:\n"
        for question in materials['practice_questions']:
            formatted_output += f"- {question}\n"

        return formatted_output

class ChatSession:
    def __init__(self, materials, ollama_model):
        self.materials = materials
        self.ollama_model = ollama_model
        self.vector_embeddings = self._generate_embeddings(materials)
        logger.info("Chat session initialized with model: %s", self.ollama_model)
        logger.info("Size of embeddings: %s", self.vector_embeddings.shape)

    def _generate_embeddings(self, materials):
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(materials)
        logger.info("Generated embeddings for materials")
        return embeddings

    async def ask_question(self, question):
        try:
            message = {'role': 'user', 'content': question}
            client = ollama.AsyncClient()
            response = await client.chat(model=self.ollama_model, messages=[message], stream=True)
            answer = ""
            async for part in response:
                answer += part['message']['content']
            logger.info("Answer: %s", answer)
            return answer
        except ollama.ResponseError as e:
            logger.error("Error in response: %s", e)
            st.error(f"Error in response: {e}")
            return "Error in response."

    async def start_chat(self):
        st.write("Welcome to the interactive Q&A session!")
        question = st.text_input("Ask a question about your study materials (or type 'exit' to quit):")
        if question.lower() == 'exit':
            st.write("Ending the chat session.")
            return
        answer = await self.ask_question(question)
        st.write(f"Answer: {answer}")

def main():
    st.title("OCR Study Guide Assistant")

    # Parent-level navigation
    st.sidebar.title("Manage Study Guides")
    action = st.sidebar.selectbox("Select Action", ["Select Study Guide", "Create New Study Guide", "Delete Study Guide"])

    if action == "Create New Study Guide":
        new_study_guide = st.sidebar.text_input("Enter New Study Guide Name")
        if st.sidebar.button("Create"):
            if new_study_guide:
                os.makedirs(os.path.join("study_guides", new_study_guide), exist_ok=True)
                st.sidebar.success(f"Study guide '{new_study_guide}' created.")
                st.session_state["selected_study_guide"] = new_study_guide
            else:
                st.sidebar.error("Please enter a name for the new study guide.")

    elif action == "Delete Study Guide":
        study_guides = [d for d in os.listdir("study_guides") if os.path.isdir(os.path.join("study_guides", d))]
        study_guide_to_delete = st.sidebar.selectbox("Select Study Guide to Delete", study_guides)
        if st.sidebar.button("Delete"):
            if study_guide_to_delete:
                study_guide_dir = os.path.join("study_guides", study_guide_to_delete)
                for root, dirs, files in os.walk(study_guide_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(study_guide_dir)
                st.sidebar.success(f"Study guide '{study_guide_to_delete}' deleted.")
            else:
                st.sidebar.error("Please select a study guide to delete.")

    elif action == "Select Study Guide":
        study_guides = [d for d in os.listdir("study_guides") if os.path.isdir(os.path.join("study_guides", d))]
        selected_study_guide = st.sidebar.selectbox("Select Study Guide", study_guides)
        if selected_study_guide:
            st.session_state["selected_study_guide"] = selected_study_guide

    # Ensure the study_guides directory exists
    os.makedirs("study_guides", exist_ok=True)

    # Display selected study guide contents
    if "selected_study_guide" in st.session_state:
        selected_study_guide = st.session_state["selected_study_guide"]
        st.sidebar.subheader(f"Contents of '{selected_study_guide}'")
        study_guide_dir = os.path.join("study_guides", selected_study_guide)
        extracted_texts = []
        image_files = []
        for file in os.listdir(study_guide_dir):
            if file.endswith(".txt"):
                with open(os.path.join(study_guide_dir, file), "r") as f:
                    text = f.read()
                    extracted_texts.append(text)
            elif file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(file)

        # Display images as thumbnails
        if image_files:
            st.sidebar.subheader("Images")
            for image_file in image_files:
                image_path = os.path.join(study_guide_dir, image_file)
                image = Image.open(image_path)
                st.sidebar.image(image, caption=image_file, use_column_width=False, width=100)

        if st.sidebar.button("Generate Study Materials"):
            generator = MaterialGenerator(output_file=os.path.join(study_guide_dir, 'materials.json'))
            materials = generator.generate_materials(extracted_texts)
            st.sidebar.success("Study materials generated.")

        if st.sidebar.button("Start Study Session"):
            generator = MaterialGenerator(output_file=os.path.join(study_guide_dir, 'materials.json'))
            materials = generator.load_materials()
            if materials:
                st.subheader("Study Session")
                st.text(generator.format_materials(materials))
                chat_session = ChatSession(materials, "orca-mini")
                asyncio.run(chat_session.start_chat())
            else:
                st.sidebar.warning("No study materials found. Please generate study materials first.")

    # Create or select a study guide
    if "selected_study_guide" in st.session_state:
        study_guide = st.session_state["selected_study_guide"]
        st.header(f"Study Guide: {study_guide}")

        uploaded_file = st.file_uploader("Upload an image file for OCR analysis", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            os.makedirs("temp", exist_ok=True)

            with open(os.path.join("temp", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
                image_path = f.name

            image = Image.open(image_path)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Run OCR"):
                ocr_processor = OCRProcessor()
                result = ocr_processor.perform_ocr(image_path)
                if result:
                    st.subheader("OCR Recognition Result:")
                    st.text(result.replace("\n", " "))
                    ocr_processor.save_response(study_guide, uploaded_file.name, result)
                    st.success(f"Response saved to study guide '{study_guide}'")
                    st.session_state["last_uploaded_image"] = image_path

        # Display thumbnails of images in the page below the study guide title
        if "last_uploaded_image" not in st.session_state:
            st.subheader("Images in Study Guide")
            cols = st.columns(4)  # Create 4 columns for displaying images
            for idx, image_file in enumerate(image_files):
                image_path = os.path.join(study_guide_dir, image_file)
                image = Image.open(image_path)
                with cols[idx % 4]:  # Cycle through columns
                    st.image(image, caption=image_file, use_column_width=True)

        # Display log messages
        st.subheader("Event Log")
        if 'log_messages' in st.session_state:
            st.text_area("Logs", value="\n".join(st.session_state['log_messages']), height=200)

if __name__ == "__main__":
    main()