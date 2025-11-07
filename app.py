from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# -------------------------
# Load NLLB Translation Model
# -------------------------
MODEL_NAME = "facebook/nllb-200-distilled-600M"
print("⏳ Loading NLLB model... This may take a few minutes.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print("✅ Model loaded successfully!\n")

# -------------------------
# Supported Languages
# -------------------------
languages = {
    "English": "eng_Latn", "Tamil": "tam_Taml", "Hindi": "hin_Deva", "Telugu": "tel_Telu",
    "Malayalam": "mal_Mlym", "Kannada": "kan_Knda", "Bengali": "ben_Beng", "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr", "Punjabi": "pan_Guru", "Odia": "ory_Orya", "Urdu": "urd_Arab",
    "French": "fra_Latn", "Spanish": "spa_Latn", "German": "deu_Latn", "Italian": "ita_Latn",
    "Russian": "rus_Cyrl", "Japanese": "jpn_Jpan", "Korean": "kor_Hang", "Chinese (Simplified)": "zho_Hans",
    "Arabic": "arb_Arab", "Turkish": "tur_Latn", "Indonesian": "ind_Latn", "Vietnamese": "vie_Latn",
    "Thai": "tha_Thai", "Portuguese": "por_Latn", "Dutch": "nld_Latn", "Swedish": "swe_Latn",
    "Greek": "ell_Grek", "Hebrew": "heb_Hebr", "Polish": "pol_Latn", "Czech": "ces_Latn",
    "Romanian": "ron_Latn", "Finnish": "fin_Latn", "Ukrainian": "ukr_Cyrl", "Hungarian": "hun_Latn"
}

# -------------------------
# Translation Function
# -------------------------
def translate_text(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang)
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    translated_text = ""
    src_sel = "English"
    tgt_sel = "Tamil"
    text = ""

    if request.method == "POST":
        src_sel = request.form["source_lang"]
        tgt_sel = request.form["target_lang"]
        text = request.form["text"]
        translated_text = translate_text(text, languages[src_sel], languages[tgt_sel])

    return render_template("index.html",
                           languages=languages,
                           translated_text=translated_text,
                           src_sel=src_sel,
                           tgt_sel=tgt_sel,
                           text=text)

if __name__ == "__main__":
    app.run(debug=True)
