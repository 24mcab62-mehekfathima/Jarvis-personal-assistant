import os
import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
DATA_DIR = "notes"
DB_PATH = "db.json"

# Load model locally
model = SentenceTransformer('all-MiniLM-L6-v2')  # fast & small

# Load or create database
if Path(DB_PATH).exists():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)
else:
    db = []

# Read notes
notes = []
if not Path(DATA_DIR).exists():
    raise FileNotFoundError(f"The folder '{DATA_DIR}' does not exist")

for file in Path(DATA_DIR).glob("*.txt"):
    content = file.read_text(encoding="utf-8").strip()
    if content:
        notes.append({"filename": file.name, "text": content})

print(f"Loaded {len(notes)} notes from '{DATA_DIR}' folder.")

# Create embeddings locally
for note in tqdm(notes, desc="Embedding notes"):
    if any(d["filename"] == note["filename"] for d in db):
        continue

    embedding = model.encode(note["text"]).tolist()
    db.append({
        "filename": note["filename"],
        "text": note["text"],
        "embedding": embedding
    })
    print(f"✅ Processed: {note['filename']}")

# Save database
with open(DB_PATH, "w", encoding="utf-8") as f:
    json.dump(db, f, indent=4, ensure_ascii=False)

print(f"\nAll notes embedded and saved to '{DB_PATH}'")