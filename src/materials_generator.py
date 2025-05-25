from sentence_transformers import SentenceTransformer
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from collections import Counter
import json
import os
from streamlit_logger import get_logger

class MaterialGenerator:
    def __init__(self, output_file='materials.json'):
        self.logger = get_logger("streamlit-logger")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        self.qg_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        self.qg_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
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
                    self.logger.info("Generated summary for text")
                    summaries.append(summary[0]['summary_text'])
                else:
                    self.logger.warning("Summarizer returned an empty result")
                    summaries.append("Summary not available.")
            except Exception as e:
                self.logger.error("Error generating summary: %s", e)
                summaries.append("Summary not available.")
        return summaries

    def create_vocabulary_list(self, texts):
        words = ' '.join(texts).split()
        vocab_counter = Counter(words)
        vocab_list = [(word, count) for word, count in vocab_counter.most_common()]
        self.logger.info("Generated vocabulary list")
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
                self.logger.info("Generated practice questions for text")
            except Exception as e:
                self.logger.error("Error generating practice questions: %s", e)
                questions.append("Could not generate questions.")
        return questions

    def save_materials(self, materials):
        with open(self.output_file, 'w') as file:
            json.dump(materials, file)
            self.logger.info("Materials saved to %s", self.output_file)

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
