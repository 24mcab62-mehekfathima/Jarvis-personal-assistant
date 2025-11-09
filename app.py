import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import gradio as gr

# ---------------- LOAD MODEL ----------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------- LOAD NOTES DATABASE ----------------
with open("db.json", "r", encoding="utf-8") as f:
    db = json.load(f)

# ---------------- HELPER FUNCTION ----------------
def find_relevant_note(query):
    query_emb = model.encode(query).reshape(1, -1)
    embeddings = [np.array(d["embedding"]) for d in db]
    sims = cosine_similarity(query_emb, embeddings)[0]
    idx = np.argmax(sims)
    return db[idx]["text"]

# ---------------- GRADIO FUNCTION ----------------
def jarvis_ui(user_input):
    return find_relevant_note(user_input)

# ---------------- GRADIO UI ----------------
with gr.Blocks(css="""
    body { background-color: #121212; color: #ffffff; font-family: 'Helvetica Neue', Arial, sans-serif; }
    .gr-button { background-color: #1F1F1F; color: #ffffff; border: 1px solid #333; border-radius:8px; padding:10px; font-size:16px; }
    .gr-input, .gr-textbox { background-color: #1F1F1F; color: #ffffff; border: 1px solid #333; border-radius:8px; font-size:16px; padding:10px; }
    .gr-label { font-weight: bold; color: #ffffff; font-size: 18px; }
    #header { text-align:center; font-size: 32px; font-weight:bold; margin-bottom:5px; }
    #subheader { text-align:center; font-size:18px; color: lightgray; margin-bottom:25px; }
""") as demo:

    gr.Markdown("Jarvis — Your Personal Assistant", elem_id="header")
    gr.Markdown("Ask questions and get answers directly from your uploaded notes.", elem_id="subheader")

    with gr.Row():
        with gr.Column(scale=1):
            user_question = gr.Textbox(
                label="Your Question",
                placeholder="Type your question here...",
                lines=4
            )
            submit_btn = gr.Button("Ask Jarvis")
        with gr.Column(scale=1):
            jarvis_answer = gr.Textbox(
                label="Jarvis Answer",
                placeholder="Answer will appear here...",
                lines=15
            )

    submit_btn.click(jarvis_ui, inputs=user_question, outputs=jarvis_answer)

# ---------------- LAUNCH ----------------
# Let Gradio pick a free port automatically to avoid port conflicts
demo.launch(share=True, debug=True)