import streamlit as st
import pandas as pd
from transformers import TFBertForSequenceClassification, BertTokenizer
from Bio import SeqIO
from io import StringIO
import tensorflow as tf

# ----------------------
# 1. PAGE CONFIGURATION
# ----------------------
st.set_page_config(
    page_title="ASAMP – Antimicrobial Peptide Predictor",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #f4f7f9; }
.stButton>button {
    width: 100%;
    border-radius: 6px;
    height: 3em;
    background-color: #004a99;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ----------------------
# 2. LOAD MODEL FROM HF
# ----------------------
HF_MODEL_ID = "Pharmson/temp-pharmson-weights-beta"

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(HF_MODEL_ID)
    model = TFBertForSequenceClassification.from_pretrained(HF_MODEL_ID)
    return model, tokenizer

model, tokenizer = load_model()

# ----------------------
# 3. PREDICTION FUNCTION
# ----------------------
def predict_sequence(seq: str):
    seq = seq.strip().replace(" ", "")
    spaced_seq = " ".join(list(seq))

    inputs = tokenizer(
        spaced_seq,
        return_tensors="tf",
        padding=True,
        truncation=True
    )

    outputs = model(**inputs)
    probs = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]

    if probs[1] > probs[0]:
        return "AMP", float(probs[1])
    else:
        return "Non-AMP", float(probs[0])

# ----------------------
# 4. STREAMLIT UI
# ----------------------
st.title("ASAMP")
st.subheader("AntiMicrobial Peptide Predictor")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# ---- Single Prediction ----
with tab1:
    seq = st.text_input("Enter amino acid sequence", placeholder="KLLKLLK")

    if st.button("Predict"):
        if len(seq) < 5:
            st.warning("Sequence must be at least 5 amino acids long.")
        else:
            with st.spinner("Analyzing sequence..."):
                pred, conf = predict_sequence(seq)
                if pred == "AMP":
                    st.success(f"**Result: AMP** (Confidence: {conf:.2%})")
                else:
                    st.error(f"**Result: Non-AMP** (Confidence: {conf:.2%})")

# ---- Batch Prediction ----
with tab2:
    st.write("Paste FASTA sequences or upload a FASTA file.")

    input_type = st.radio("Input method", ["Paste FASTA", "Upload file"])
    sequences, names = [], []

    if input_type == "Paste FASTA":
        pasted = st.text_area("Paste FASTA", height=200)
        if pasted:
            for record in SeqIO.parse(StringIO(pasted), "fasta"):
                sequences.append(str(record.seq))
                names.append(record.id)
    else:
        file = st.file_uploader("Upload FASTA file", type=["fasta", "faa", "txt"])
        if file:
            for record in SeqIO.parse(StringIO(file.getvalue().decode()), "fasta"):
                sequences.append(str(record.seq))
                names.append(record.id)

    if st.button("Analyze All"):
        if not sequences:
            st.error("No sequences found.")
        else:
            results = []
            progress = st.progress(0)

            for i, (seq, name) in enumerate(zip(sequences, names)):
                try:
                    pred, conf = predict_sequence(seq)
                    results.append({
                        "ID": name,
                        "Sequence": seq,
                        "Prediction": pred,
                        "Confidence": round(conf, 4)
                    })
                except:
                    results.append({
                        "ID": name,
                        "Sequence": seq,
                        "Prediction": "Error",
                        "Confidence": 0.0
                    })

                progress.progress((i + 1) / len(sequences))

            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download Results",
                df.to_csv(index=False),
                "asamp_predictions.csv",
                "text/csv"
            )
