from setuptools import setup, find_packages

setup(
    name='study-materials-generator',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project to generate study materials and interactive Q&A chat sessions for high school students based on their notes and quiz materials.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pytesseract',  # For OCR tasks
        'opencv-python',  # For image processing
        'numpy',  # For numerical operations
        'scikit-learn',  # For vectorization and machine learning tasks
        'flask',  # For creating a web interface (if needed)
        # Add other dependencies as required
    ],
    entry_points={
        'console_scripts': [
            'study-materials-generator=main:main',  # Adjust according to your main function location
        ],
    },
)