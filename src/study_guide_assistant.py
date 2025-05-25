
import os
import streamlit as st
from PIL import Image
import asyncio
from ocr_processor import OCRProcessor
from material_generator import MaterialGenerator
from chat_session import ChatSession

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
