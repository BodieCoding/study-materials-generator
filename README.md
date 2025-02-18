# Study Materials Generator

This project is designed to help high school students generate study materials and engage in interactive Q&A chat sessions based on their notes and quiz materials. It leverages OCR capabilities and other functionalities provided by the Ollama installation.

## Project Structure

```
study-materials-generator
├── src
│   ├── main.py                # Entry point of the application
│   ├── ocr
│   │   └── ocr_processor.py    # Handles OCR tasks
│   ├── chat
│   │   └── chat_session.py      # Implements interactive Q&A chat
│   ├── materials
│   │   └── material_generator.py # Generates study materials
│   └── utils
│       └── helpers.py          # Utility functions
├── requirements.txt            # Project dependencies
├── setup.py                    # Setup configuration
└── README.md                   # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/BodieCoding/study-materials-generator
   cd study-materials-generator
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure that you have the Ollama installation set up on your local machine. Follow the instructions provided in the Ollama documentation for installation and configuration.

## Usage

1. Run the application:
   ```
   python src/main.py
   ```

2. Follow the prompts to specify the directory containing your notes and quiz materials.

3. The application will initiate the OCR process to extract text from your files.

4. Once the text is extracted, you can generate study materials such as summaries, vocabulary lists, and practice questions.

5. Start an interactive Q&A chat session to ask questions about the generated study materials.

## Configuration

You may need to specify the Ollama model for OCR and other configurations in the `src/main.py` file. Make sure to adjust the settings according to your requirements.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.