import os
import json
import asyncio
from PIL import Image
import PyPDF2
import aiohttp
import ollama
import base64
import io
import logging

class OcrProcessor:
    def __init__(self, ollama_model, session_type ,manifest_file='manifest.json'):
        self.ollama_model = ollama_model
        self.session_type = session_type
        self.manifest_file = manifest_file
        self.manifest = self.load_manifest()
        logging.info("OCR Processor initialized with model: %s", ollama_model)

    def load_manifest(self):
        if os.path.exists(self.manifest_file):
            with open(self.manifest_file, 'r') as file:
                logging.info("Loading manifest from %s", self.manifest_file)
                return json.load(file)
        return {}

    def save_manifest(self):
        with open(self.manifest_file, 'w') as file:
            json.dump(self.manifest, file)
            logging.info("Manifest saved to %s", self.manifest_file)

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
        logging.info("Processed files in directory: %s", directory)
        return extracted_text

    async def process_png(self, file_path):
        logging.info("Processing PNG file: %s", file_path)
        text = await self.extract_text_from_image(file_path)
        return text

    async def process_pdf(self, file_path):
        logging.info("Processing PDF file: %s", file_path)
        text = self.extract_text_from_pdf(file_path)
        return text

    PROMPT_EXTRACT_TEXT = "Please quickly extract text from this image."
    PROMPT_EXTRACT_TEXT_FROM_PDF = "Please quickly extract text from this PDF."
    PROMPT_START_CHAT = f' You are an AI Assistant for a study session. I will provide you with a set of learning materials (e.g., text, articles, code snippets).
                        Analyze these materials and generate a set of comprehension questions covering key concepts.
                        Present these questions to the "User" (me) one by one.
                        Evaluate my answers and provide feedback (e.g., "Correct!", "Close, but...", "Lets explore this further").
                        If I answer incorrectly, offer explanations, examples, or additional resources to help me understand the concept better.
                        Repeat this process until I successfully answer all questions correctly.
                        Be patient and encouraging throughout the session.'
    
    async def extract_text_from_image(self, file_path):
        image_base64 = self.image_to_base64(file_path)
        response = ollama.chat(
            model=self.ollama_model,
            messages=[{
                "role": "user",
                "content": self.PROMPT_EXTRACT_TEXT,
                "images": [image_base64]
            }]
        )
        text = response['message']['content'].strip()
        logging.info("Extracted text from image: %s", file_path)
        print(text)
        return text

    def extract_text_from_pdf(self, file_path):
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfFileReader(file)
            for page_num in range(reader.numPages):
                page = reader.getPage(page_num)
                text += page.extract_text()
        logging.info("Extracted text from PDF: %s", file_path)
        return text

    def convert_to_text(self, content):
        return content.strip()

    def image_to_base64(self, image_path):
        # Open the image file
        with Image.open(image_path) as img:
            # Create a BytesIO object to hold the image data
            buffered = io.BytesIO()
            # Save the image to the BytesIO object in a specific format (e.g., PNG)
            img.save(buffered, format="PNG")
            # Get the byte data from the BytesIO object
            img_bytes = buffered.getvalue()
            # Encode the byte data to base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            logging.info("Converted image to base64: %s", image_path)
            return img_base64
