# app.py (UI/UX improved) - Chanikya Plagiarism Checker
import streamlit as st
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import PyPDF2
import pandas as pd
import numpy as np
from io import BytesIO
# ----------------------- Page config & custom CSS -----------------------

st.set_page_config(page_title="Chanikya Plagiarism Checker", layout="wide", initial_sidebar_state="expanded")

# ----------------------- Header -----------------------

col1, col2 = st.columns([3,1])
with col1:
    st.image("logo.png", width=199)
    st.title("Chanikya Plagiarism Checker")
    st.markdown("<div class='small'>Simple, fast plagiarism checking for academic content — upload TXT, DOCX, PDF</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div style='text-align:right'><span class='badge blue'>Demo</span><br><span class='small'>v1.0</span></div>", unsafe_allow_html=True)
# ----------------------- Sidebar (controls, help, dataset) -----------------------
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload files (txt / docx / pdf)", accept_multiple_files=True, type=["txt","docx","pdf"])
    threshold = st.slider("Flag threshold (%)", 10, 100, 70, 1)
    show_sentences = st.checkbox("Show similar sentence pairs", value=True)
    st.markdown("---")
    st.header("Quick actions")
    if st.button("Load sample dataset (kaggle_dataset)"):
        st.session_state.load_samples = True
    st.markdown("**Tips:**\n- Use 3–8 files for fast demo\n- Short paragraphs work best for demo")
    st.markdown("---")
    st.header("About")
    st.markdown("**Chanikya Plagiarism Checker**\n\nBuilt with Streamlit + scikit-learn.\nSimple UI for college demos.")
    st.markdown("")
# ----------------------- Optional sample screenshots -----------------------
# local screenshot paths
s1 = "/mnt/data/9aad0a13-7caf-4644-924c-f80bf8c1766f.png"
s2 = "/mnt/data/3b33e639-6988-4350-9dcb-f2b59d3f29d7.png"
s3 = "/mnt/data/a753fbdd-4302-4fe3-a21f-a8e0e6097f38.png"

imgs = [p for p in (s1, s2, s3) if os.path.exists(p)]
if imgs:
    st.markdown("### Screenshots / Demo Preview")
    cols = st.columns(len(imgs))
    for c, img in zip(cols, imgs):
        c.image(img, caption=os.path.basename(img), use_column_width=True)

st.markdown("---")
# ----------------------- Helper functions -----------------------
def read_txt_stream(f):
    raw = f.read()
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except:
            try:
                return raw.decode("latin-1")
            except:
                return raw.decode(errors='ignore')
    return str(raw)

def read_docx_stream(f):
    if Document is None:
        st.error("python-docx not installed. Install with: pip install python-docx")
        return ""
    tmp = "temp_docx.docx"
    with open(tmp, "wb") as out:
        out.write(f.read())
    doc = Document(tmp)
    full = [p.text for p in doc.paragraphs]
    try:
        os.remove(tmp)
    except:
        pass
    return "\n".join(full)

def read_pdf_stream(f):
    if PyPDF2 is None:
        st.error("PyPDF2 not installed. Install with: pip install PyPDF2")
        return ""
    reader = PyPDF2.PdfReader(f)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except:
            pages.append("")
    return "\n".join(pages)

def preprocess(text):
    if text is None:
        return ""
    text = text.lower().replace("\n", " ").strip()
    return text

def compute_combined_tfidf(docs):
    vec_word = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    vec_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))
    tf_word = vec_word.fit_transform(docs)
    tf_char = vec_char.fit_transform(docs)
    from scipy.sparse import hstack
    combined = hstack([tf_word, tf_char])
    sim = cosine_similarity(combined)
    return sim

def top_similar_sentences(a_text, b_text, topn=5):
    a_sents = [s.strip() for s in a_text.split('.') if s.strip()]
    b_sents = [s.strip() for s in b_text.split('.') if s.strip()]
    pairs = []
    for sa in a_sents:
        for sb in b_sents:
            ratio = SequenceMatcher(None, sa, sb).ratio()
            pairs.append((ratio, sa, sb))
    pairs.sort(reverse=True, key=lambda x: x[0])
    return pairs[:topn]

# ----------------------- Load sample dataset (optional) -----------------------
if st.session_state.get("load_samples", False):
    # try to load files from kaggle_dataset folder if exists
    sample_dir = "kaggle_dataset"
    if os.path.exists(sample_dir):
        st.success("Loaded files from kaggle_dataset/")
        # mimic upload by creating 'uploaded' list from file paths
        uploaded = []
        for fname in sorted(os.listdir(sample_dir))[:8]:
            path = os.path.join(sample_dir, fname)
            uploaded.append(open(path, "rb"))
    else:
        st.warning("kaggle_dataset folder not found in project directory.")

# ----------------------- Main processing & UI -----------------------
if uploaded and len(uploaded) >= 2:
    # read files
    names = []
    texts = []
    progress = st.progress(0)
    total = len(uploaded)
    for idx, f in enumerate(uploaded, start=1):
        name = getattr(f, "name", f.name if hasattr(f, "name") else f"file_{idx}")
        names.append(name)
        if name.lower().endswith(".txt"):
            txt = read_txt_stream(f)
        elif name.lower().endswith(".docx"):
            txt = read_docx_stream(f)
        elif name.lower().endswith(".pdf"):
            txt = read_pdf_stream(f)
        else:
            txt = ""
        texts.append(preprocess(txt))
        progress.progress(int(idx/total * 100))
    progress.empty()

    # compute similarity
    with st.spinner("Computing combined TF-IDF & similarity..."):
        sim = compute_combined_tfidf(texts)

    # build table
    rows = []
    n = len(names)
    for i in range(n):
        for j in range(i+1, n):
            score = float(sim[i,j]) * 100.0
            rows.append({"Doc A": names[i], "Doc B": names[j], "Similarity (%)": round(score,2)})
    df = pd.DataFrame(rows).sort_values("Similarity (%)", ascending=False).reset_index(drop=True)

    # dashboard summary cards
    c1, c2, c3 = st.columns(3)
    c1.markdown("<div class='card'><b>Total files</b><div class='small'>%d files uploaded</div></div>" % n, unsafe_allow_html=True)
    top_sim = df["Similarity (%)"].max() if not df.empty else 0
    c2.markdown("<div class='card'><b>Top similarity</b><div class='small'>%s%%</div></div>" % round(top_sim,2), unsafe_allow_html=True)
    flagged_count = len(df[df["Similarity (%)"] >= threshold])
    c3.markdown("<div class='card'><b>Flagged pairs</b><div class='small'>%d pairs (>%d%%)</div></div>" % (flagged_count, threshold), unsafe_allow_html=True)

    st.subheader("Pairwise Similarity")
    st.dataframe(df, use_container_width=True)

    st.markdown("### Flagged Pairs (evidence)")
    flagged = df[df["Similarity (%)"] >= threshold]
    if flagged.empty:
        st.info("No pairs exceeded the threshold.")
    else:
        for idx, row in flagged.iterrows():
            a = row["Doc A"]; b = row["Doc B"]; score = row["Similarity (%)"]
            st.markdown(f"<div class='card'><b>{a}</b>  —  <b>{b}</b>  <span style='float:right' class='badge red'>{score}%</span></div>", unsafe_allow_html=True)
            if show_sentences:
                i = names.index(a); j = names.index(b)
                top = top_similar_sentences(texts[i], texts[j], topn=5)
                for r, sa, sb in top:
                    st.markdown(f"> **Ratio:** {r:.2f}  \n> A: {sa}  \n> B: {sb}")
                st.markdown("---")

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download similarity report (CSV)", csv, file_name="similarity_report.csv", mime="text/csv")

else:
    st.info("Upload at least 2 files (txt / docx / pdf) to compare. Use the left sidebar to upload files or load sample dataset.")
    st.markdown("""
    **Demo instructions (simple):**
    1. Click 'Browse files' and select multiple files (txt/docx/pdf).  
    2. Set the threshold slider (default 70%).  
    3. upload 'Load sample dataset'  to quickly test.  
    4. Wait for the app to compute and then view results table and flagged pairs.
    """)
