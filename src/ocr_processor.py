import streamlit as st
import base64
import aiohttp
from PIL import Image
import os
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import asyncio
import nest_asyncio
import PyPDF2
from streamlit_logger import get_logger

class OCRProcessor:
    def __init__(self):
        self.logger = get_logger()
        nest_asyncio.apply()

    async def process_study_guide(self, study_guide_name):      
        try:
            study_guide_dir = os.path.join("study_guides", study_guide_name)

            studyguide_manifest_name = f"manifest-{study_guide_name}.json"
            studyguide_manifest_location = os.path.join(study_guide_dir, studyguide_manifest_name)
            studyguide_manifest_contents = {}
            extracted_text = []
            tasks = []
            aggregate_encoded_image_tasks = []
            base64_images_to_process = []

            # Ensure the directory exists
            if not os.path.exists(study_guide_dir):
                os.makedirs(study_guide_dir, exist_ok=True)

            # Load the manifest file if it exists
            if os.path.exists(studyguide_manifest_location):
                with open(studyguide_manifest_location, "r") as file:
                    studyguide_manifest_contents = json.load(file) 

            # Display the contents of the manifest on the Streamlit screen
            st.write("Study Guide Manifest Contents:")
            st.json(studyguide_manifest_contents)
            
            # Walk through the study guide directory
            for root, _, files in os.walk(study_guide_dir):
                # Process each file in the directory
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', 'pdf')):
                        st.info(f"Checking if file already processed: {file_path}")
                        # Skip files that have already been processed
                        if studyguide_manifest_contents.get(file_path, False):
                            st.info(f"Skipping file: {file_path}")
                            continue
                    # Check if the file is an image
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                        # Add task to base64 encode the image
                        st.info(f"Processing image: {file_path}")
                        aggregate_encoded_image_tasks.append(self.encode_image_to_base64(file_path))
                        # Mark the image as processed in the manifest
                        studyguide_manifest_contents[file_path] = True
                    # Check if the file is a PDF
                    elif file.lower().endswith('.pdf'):
                        # Add task to process the PDF file
                        tasks.append(self.process_pdf(file_path))
                        # Mark the PDF as processed in the manifest
                        studyguide_manifest_contents[file_path] = True
            # Check if there are any image processing tasks
            if aggregate_encoded_image_tasks:
                # Wait for all images to be base64 encoded, and gather the results as a list
                base64_images_to_process = await asyncio.gather(*aggregate_encoded_image_tasks)
                # Check if there are any base64 encoded images in the list to process
                if base64_images_to_process:
                    # Add the task to extract text from images
                    tasks.append(self.extract_text_from_images(base64_images_to_process))
            # Wait for all tasks to complete
            if tasks:
                # Gather text from image and pdf extraction tasks
                extracted_text.extend(await asyncio.gather(*tasks))
            # Save the manifest file
            with open(studyguide_manifest_location, "w") as file:
                json.dump(studyguide_manifest_contents, file)

            # Append the extracted text to the ocr file
            extracted_text_location = os.path.join(study_guide_dir, f"ocr-{study_guide_name}.txt")
            with open(extracted_text_location, "a") as f:
                f.write("\n".join(extracted_text))

            return extracted_text 
        except Exception as e:
            st.error(f"Error processing study guide: {e}")
            return None
    
    async def extract_text_from_images(self, base64_images):
        st.info("Extracting text from images...")
        base64_images = await self.ensure_list(base64_images)
        payload = await self.create_payload(base64_images)
        response = await self.send_request(payload)
        return await self.process_response(response)

    async def ensure_list(self, base64_images):
        if not isinstance(base64_images, list):
            base64_images = [base64_images]
        return base64_images
    
    async def create_payload(self, base64_images):
        prompt = """You are an advanced OCR tool. Your task is to accurately transcribe the text from the provided image.
        Please follow these guidelines to ensure the transcription is as correct as possible:
        1. **Preserve Line Structure**: Transcribe the text exactly as it appears in the image, including line breaks.
        2. **Avoid Splitting Words**: Ensure that words are fully formed, even if they appear across two lines or are partially obscured. Join word parts together appropriately to create complete words.
        3. **Correct Unnatural Spacing**: Do not add extra spaces between characters in a word. Make sure the words are spaced naturally, without any unwanted breaks or gaps.
        4. **Recognize and Correct Word Breaks**: If any word is mistakenly broken into parts, join them correctly to produce a natural, readable word. 
        5. **Output as a Block of Text**: Output the entire transcribed text as a block, maintaining the line breaks, but ensuring that each word appears as it should, with correct spelling, no character-level splits, and no hyphenations unless they appear naturally in the image."""
      
        payload = {
            "model": "llama3.2-vision",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": base64_images
                }
            ]
        }
        return payload

    async def send_request(self, payload):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post("http://localhost:11434/api/chat", 
                    headers={"Content-Type": "application/json"}, 
                    json=payload) as response:
                        return await self.process_response(response)
        except aiohttp.ClientError as e:
            st.error(f"Error sending request to OCR tool: {e}")
            return None

    async def process_response(self, response):
        try:
            if response and response.status == 200:
                response_json = await response.json()
                choices = response_json.get("choices", [])
                if choices and "message" in choices[0] and "content" in choices[0]["message"]:
                    return choices[0]["message"]["content"].strip()
                else:
                    return "No text found in the image."
            else:
                st.error(f"Error: {response.status} - {await response.text()}")
                return None
        except (Exception, json.JSONDecodeError) as e:
            st.error(f"Error extracting text from images: {e}")
            st.write("Raw Response:", await response.text())
            return None

    async def process_pdf(self, file_path):
        self.logger.info("Processing PDF file: %s", file_path)
        text = await self.extract_text_from_pdf(file_path)
        return text
    
    async def extract_text_from_pdf(self, file_path):
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text
    
    async def encode_image_to_base64(self, image_path):
        """Convert an image file to a base64 encoded string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

