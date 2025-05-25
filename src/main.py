import streamlit as st
import base64
import requests
from PIL import Image
import os
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import asyncio
import nest_asyncio
import ollama
import PyPDF2
from streamlit_logger import get_logger
from chat_session import ChatSession
from materials_generator import MaterialGenerator
from ocr_processor import OCRProcessor

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Create a new event loop and set it as default
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

#study guide selection and creation enum
class StudyGuideAction:
    SELECT_STUDY_GUIDE = "Select Study Guide"
    CREATE_NEW_STUDY_GUIDE = "Create New Study Guide"
    DELETE_STUDY_GUIDE = "Delete Study Guide"  


async def main():

    displayed_study_guide = st.session_state.get("displayed_study_guide", None)
    selected_study_guide = st.session_state.get("selected_study_guide", None)

    st.title("OCR Study Guide Assistant")

    # Ensure the study_guides directory exists
    os.makedirs("study_guides", exist_ok=True)

    # Parent-level navigation
    st.sidebar.title("Manage Study Guides")
    study_guide_action_sel = st.sidebar.selectbox("Study Guide Action", [StudyGuideAction.SELECT_STUDY_GUIDE, StudyGuideAction.CREATE_NEW_STUDY_GUIDE, StudyGuideAction.DELETE_STUDY_GUIDE])

    if study_guide_action_sel == StudyGuideAction.CREATE_NEW_STUDY_GUIDE:
        new_study_guide = st.sidebar.text_input("Enter New Study Guide Name")
        if st.sidebar.button("Create"):
            if new_study_guide:
                os.makedirs(os.path.join("study_guides", new_study_guide), exist_ok=True)
                st.sidebar.success(f"Study guide '{new_study_guide}' created.")
                st.session_state["selected_study_guide"] = new_study_guide
                st.session_state["displayed_study_guide"] = new_study_guide
                displayed_study_guide = new_study_guide
                selected_study_guide = new_study_guide
            else:
                st.sidebar.error("Please enter a name for the new study guide.")

    elif study_guide_action_sel == StudyGuideAction.DELETE_STUDY_GUIDE:
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

    elif study_guide_action_sel == StudyGuideAction.SELECT_STUDY_GUIDE:
        study_guides = [d for d in os.listdir("study_guides") if os.path.isdir(os.path.join("study_guides", d))]
        selected_study_guide = st.sidebar.selectbox("Select Study Guide", study_guides)
        if selected_study_guide:
            st.session_state["selected_study_guide"] = selected_study_guide

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
                st.sidebar.image(image, caption=image_file, use_container_width=False, width=100)

        # Check if materials.json exists and display a message
        materials_file_path = os.path.join(study_guide_dir, 'materials.json')
        if os.path.exists(materials_file_path):
            st.sidebar.info("Existing study materials found.")

        if st.sidebar.button("Generate Study Materials"):
            generator = MaterialGenerator(output_file=materials_file_path)
            materials = generator.generate_materials(extracted_texts)
            st.sidebar.success("Study materials generated.")

        if st.sidebar.button("Start Study Session") or "in_chat_session" in st.session_state:
            generator = MaterialGenerator(output_file=materials_file_path)
            materials = generator.load_materials()
            if materials:
                st.subheader("Study Session")
                # Set a flag to remember we're in a chat session
                st.session_state.in_chat_session = True
                
                # Check if chat_session exists in session_state
                if 'chat_session' not in st.session_state:
                    st.session_state.chat_session = ChatSession(materials['summaries'], "orca-mini")
                
                # Always call start_chat on reruns as long as we're in a chat session
                await st.session_state.chat_session.start_chat()
            else:
                st.sidebar.warning("No study materials found. Please generate study materials first.")

    # Create or select a study guide
    if "selected_study_guide" in st.session_state:
        extracted_texts = []
        image_files = []
        uploaded_files = None
        study_guide = st.session_state["selected_study_guide"]
        st.header(f"Study Guide: {study_guide}")

        #get the location of the study guide, create it if it doesn't exist
        study_guide_dir = os.path.join("study_guides", study_guide)
        os.makedirs(study_guide_dir, exist_ok=True)

        # Upload images or PDF files for OCR analysis
        st.subheader("Upload Files for OCR Analysis")

        uploaded_files = st.file_uploader("Upload images or PDF files for OCR analysis", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                with open(os.path.join(study_guide_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    image_path = f.name

        # Load existing artifacts
        for file in os.listdir(study_guide_dir):
            if file.endswith(".txt"):
                with open(os.path.join(study_guide_dir, file), "r") as f:
                    text = f.read()
                    extracted_texts.append(text)
            elif file.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                image_files.append(file)

        if st.button("Run OCR"):
            ocr_processor = OCRProcessor()
            extracted_texts = await ocr_processor.process_study_guide(study_guide)
            st.success("OCR processing complete.")

        # Display thumbnails of images in the page below the study guide title
        st.subheader("Images in Study Guide")
        cols = st.columns(4)  # Create 4 columns for displaying images
        for idx, image_file in enumerate(image_files):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(study_guide_dir, image_file)
                image = Image.open(image_path)
                with cols[idx % 4]:  # Cycle through columns
                    st.image(image, caption=image_file, use_container_width=True)

        # Display log messages
        st.subheader("Event Log")
        if 'log_messages' in st.session_state:
            st.text_area("Logs", value="\n".join(st.session_state['log_messages']), height=200)

if __name__ == "__main__":
    asyncio.run(main())