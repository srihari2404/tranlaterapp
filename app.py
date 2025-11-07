from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# -------------------------
# Load Multiple Models
# -------------------------
print("⏳ Loading translation models... Please wait...")

MODELS = {
    "zh-en": "Helsinki-NLP/opus-mt-zh-en",  # Chinese → English
    "en-zh": "Helsinki-NLP/opus-mt-en-zh"   # English → Chinese
}

tokenizers = {}
models = {}

for direction, name in MODELS.items():
    print(f"Loading {direction} model...")
    tokenizers[direction] = AutoTokenizer.from_pretrained(name)
    models[direction] = AutoModelForSeq2SeqLM.from_pretrained(name)

print("✅ All models loaded successfully!\n")

# -------------------------
# Supported Languages
# -------------------------
languages = {
    "Chinese → English": "zh-en",
    "English → Chinese": "en-zh"
}

# -------------------------
# Translation Function
# -------------------------
def translate_text(text, direction):
    tokenizer = tokenizers[direction]
    model = models[direction]
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_length=400)
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated

# -------------------------
# Flask Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    translated_text = ""
    text = ""
    direction = "zh-en"

    if request.method == "POST":
        text = request.form["text"]
        direction = request.form["direction"]
        translated_text = translate_text(text, direction)

    return render_template(
        "index.html",
        languages=languages,
        translated_text=translated_text,
        direction=direction,
        text=text
    )


if __name__ == "__main__":
    app.run(debug=True)
