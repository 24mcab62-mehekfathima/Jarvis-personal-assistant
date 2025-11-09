# Jarvis-personal-assistant
Project Overview:
Jarvis is a self-hosted, AI-powered personal assistant designed to answer questions using your own notes. This project demonstrates the practical use of embeddings and similarity search to build a knowledge-based AI assistant, with a professional and user-friendly interface inspired by modern dark-themed designs.

Key Features:

Answers questions based on custom notes you provide.

Uses semantic search with embeddings to find the most relevant content from your notes.

Interactive UI built with Gradio for an intuitive question-answering experience.

Dark-themed, professional design for a modern, sleek look.

Fully self-contained – no paid OpenAI API required, works with open-source models (sentence-transformers).


How It Works:

1. Prepare Notes:

Store your knowledge or study material in plain text files inside the notes/ folder.

Each file can contain topics, summaries, or any content you want Jarvis to know.



2. Ingest Notes:

The ingest.py script reads all text files in notes/, converts the content into embeddings using a local SentenceTransformer model (all-MiniLM-L6-v2), and saves the embeddings in db.json.

These embeddings allow Jarvis to semantically understand your queries and match them with the most relevant notes.



3. Query Jarvis:

The app.py script launches a professional, dark-themed UI.

Users type a question into the input box and click "Ask Jarvis."

Jarvis searches db.json using cosine similarity on embeddings to find the most relevant note.

The answer is displayed in a large, readable textbox.



4. Self-Contained & Free:

This project uses the open-source SentenceTransformer model, so it does not require paid APIs.

Works offline on your machine and retrieves answers directly from your personal notes.




How to Run:

1. Install dependencies:



pip install -r requirements.txt

2. Prepare your notes in the notes/ folder.


3. Ingest your notes to create embeddings:



python ingest.py

4. Launch Jarvis:



python app.py

5. Open the UI in your browser, type a question, and get answers instantly.



Tech Stack:

Python 3.x

Sentence Transformers (sentence-transformers) for embeddings

Numpy & Scikit-learn for similarity search

Gradio for the interactive web UI


Why This Project is Impressive for Internships:

Demonstrates practical knowledge of AI, NLP, embeddings, and semantic search.

Shows UI/UX skills with professional design.

Fully self-contained, offline, and free to run.

Highlights your ability to build real-world AI applications with open-source tools.
