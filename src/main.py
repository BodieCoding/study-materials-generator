import os
import asyncio
import logging
from ocr.ocr_processor import OcrProcessor
from materials.material_generator import MaterialGenerator
from chat.chat_session import ChatSession

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    logging.info("Welcome to the Study Materials Generator!")
    parent_directory = input("Please enter the path to your notes and quiz materials directory: ")

    if not os.path.exists(parent_directory):
        logging.error("The specified directory does not exist. Please try again.")
        return

    # Initialize OCR Processor
    logging.info("Initializing OCR Processor...")
    ocr_processor = OcrProcessor(ollama_model='llama3.2-vision')
    extracted_text = await ocr_processor.process_files_in_directory(parent_directory)
    logging.info(f"Text extracted from files:{extracted_text}")
    # Generate study materials
    logging.info("Generating study materials...")
    material_generator = MaterialGenerator()
    study_materials = material_generator.generate_materials(extracted_text)

    # Start interactive Q&A chat session
    logging.info("Starting interactive Q&A chat session...")
    chat_session = ChatSession(study_materials, ollama_model='llama3.2-vision')
    await chat_session.start_chat()

if __name__ == "__main__":
    asyncio.run(main())