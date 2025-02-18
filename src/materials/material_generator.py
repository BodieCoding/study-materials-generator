import json
import os
import logging
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from collections import Counter

class MaterialGenerator:
    def __init__(self, output_file='materials.json'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
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
        """
        Generate summaries from the extracted texts.

        Args:
            texts (list): The list of texts to summarize.

        Returns:
            list: The generated summaries.
        """
        summaries = []
        for text in texts:
            try:
                summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
                if summary and len(summary) > 0:
                    logging.info("Generated summary for text")
                    summaries.append(summary[0]['summary_text'])
                else:
                    logging.warning("Summarizer returned an empty result")
                    summaries.append("Summary not available.")
            except Exception as e:
                logging.error("Error generating summary: %s", e)
                summaries.append("Summary not available.")
        return summaries

    def create_vocabulary_list(self, texts):
        """
        Create a vocabulary list from the extracted texts.

        Args:
            texts (list): The list of texts to extract vocabulary from.

        Returns:
            list: The generated vocabulary list.
        """
        words = ' '.join(texts).split()
        vocab_counter = Counter(words)
        vocab_list = [(word, count) for word, count in vocab_counter.most_common()]
        logging.info("Generated vocabulary list")
        return vocab_list
    
    def initiate_question_answer_session(self, text):
        """
        Initiate a question-answer session with the user.

        Args:
            text (str): The text to ask questions about.

        Returns:
            list: The generated questions.
        """
        questions = []
        try:
            prompt = f' You are an AI Assistant for a study session. I will provide you with a set of learning materials (e.g., text, articles, code snippets).
                        Analyze these materials and generate a set of comprehension questions covering key concepts.
                        Present these questions to the "User" (me) one by one.
                        Evaluate my answers and provide feedback (e.g., "Correct!", "Close, but...", "Lets explore this further").
                        If I answer incorrectly, offer explanations, examples, or additional resources to help me understand the concept better.
                        Repeat this process until I successfully answer all questions correctly.
                        Be patient and encouraging throughout the session.'
        except Exception as e:
            logging.error("Error generating questions: %s", e)
        return questions

    def generate_practice_questions(self, texts):
        """
        Generate practice questions from the extracted texts.

        Args:
            texts (list): The list of texts to generate questions from.

        Returns:
            list: The generated practice questions.
        """
        # Placeholder implementation
        questions = ["What is the main idea of the text?", "List key points mentioned in the text."]
        logging.info("Generated practice questions")
        return questions

    def format_materials(self, materials):
        """
        Format the generated materials for output.

        Args:
            materials (dict): The generated materials.

        Returns:
            str: The formatted output.
        """
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

    def save_materials(self, materials):
        """
        Save the generated materials to a file.

        Args:
            materials (dict): The generated materials.
        """
        with open(self.output_file, 'w') as file:
            json.dump(materials, file)
            logging.info("Materials saved to %s", self.output_file)

    def load_materials(self):
        """
        Load the generated materials from a file.

        Returns:
            dict: The loaded materials.
        """
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r') as file:
                return json.load(file)
        return {}