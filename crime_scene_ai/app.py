# app.py
import streamlit as st
import matplotlib.pyplot as plt
import tempfile
import os
from PIL import Image

from vision import analyze_scene, scene_to_text, train_forensic_scorer
from rag import load_forensic_corpus, build_faiss_index, retrieve
from llm import generate_report

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚨 AI Crime Scene Analyzer",
    page_icon="🔍",
    layout="wide"
)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🚨 AI Crime Scene Analysis Assistant")
st.caption(
    "Multimodal RAG System · YOLOv8 Vision · Adam Optimizer · "
    "Wikipedia Forensic Knowledge · Claude LLM Reasoning"
)
st.divider()

# ── Load RAG resources (cached) ───────────────────────────────────────────────
corpus = load_forensic_corpus()
index  = build_faiss_index(corpus)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Info")
    st.metric("Forensic Knowledge Chunks", len(corpus))
    st.metric("Wikipedia Topics Loaded", 15)
    st.metric("Vector Index Type", "FAISS (cosine)")
    st.metric("Vision Model", "YOLOv8n")
    st.metric("LLM", "Claude Sonnet 4")
    st.divider()
    st.markdown("**Pipeline:**")
    st.markdown("""
    1. 🖼️ Image → YOLOv8
    2. 📝 Detections → Scene Text
    3. 📚 Scene → FAISS Retrieval
    4. 🧠 Context → Claude Report
    """)
    top_k = st.slider("RAG: Retrieved passages", 3, 8, 5)

# ── Main Layout ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Upload Crime Scene Image")
    uploaded = st.file_uploader(
        "Upload a crime scene photo",
        type=["jpg", "jpeg", "png"],
        help="Supports JPG and PNG images"
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Crime Scene", use_column_width=True)

with col2:
    if not uploaded:
        st.info("👈 Upload a crime scene image to begin analysis.")
    else:
        # ── Step 1: Vision Analysis ──────────────────────────────────────────
        with st.spinner("🔍 Running YOLOv8 object detection..."):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name

            detections = analyze_scene(tmp_path)
            os.unlink(tmp_path)
            scene_text = scene_to_text(detections)

        st.subheader("🎯 Detected Objects")
        if detections:
            for d in detections:
                st.write(f"• {d}")
        else:
            st.warning("No objects detected above confidence threshold.")

        st.subheader("📝 Scene Summary")
        st.code(scene_text, language=None)

# ── Full Width Report Section ─────────────────────────────────────────────────
if uploaded and detections is not None:
    st.divider()

    col3, col4 = st.columns([1, 1], gap="large")

    with col3:
        # ── Step 2: RAG Retrieval ────────────────────────────────────────────
        with st.spinner("📚 Retrieving forensic knowledge from Wikipedia corpus..."):
            relevant_passages = retrieve(scene_text, index, corpus, top_k=top_k)

        st.subheader("📚 Retrieved Forensic Knowledge")
        st.caption(f"Top {top_k} passages retrieved via FAISS cosine similarity")
        for i, passage in enumerate(relevant_passages, 1):
            with st.expander(f"Source {i} — {passage[:60]}..."):
                st.write(passage)

    with col4:
        # ── Step 3: LLM Report ───────────────────────────────────────────────
        with st.spinner("🧠 Claude is generating investigation report..."):
            report = generate_report(scene_text, relevant_passages)

        st.subheader("📋 Investigation Report")
        st.markdown(report)

    # ── Step 4: Adam Optimizer Visualization ─────────────────────────────────
    st.divider()
    st.subheader("📉 Adam Optimizer — Forensic Scorer Training")
    st.caption(
        "A neural network is trained to score forensic priority of detected objects. "
        "Adam optimizer (lr=0.001, β₁=0.9, β₂=0.999) with StepLR scheduling."
    )

    with st.spinner("⚡ Training forensic scorer with Adam optimizer..."):
        losses = train_forensic_scorer(epochs=30)

    col5, col6 = st.columns([2, 1])
    with col5:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(losses, color="#e63946", linewidth=2, label="Training Loss")
        ax.fill_between(range(len(losses)), losses, alpha=0.15, color="#e63946")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.set_title("Adam Optimizer Convergence on ForensicScorer")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col6:
        st.metric("Initial Loss", f"{losses[0]:.4f}")
        st.metric("Final Loss",   f"{losses[-1]:.4f}")
        st.metric("Improvement",  f"{((losses[0]-losses[-1])/losses[0]*100):.1f}%")
        st.caption("Optimizer: Adam | LR Scheduler: StepLR(γ=0.5)")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🎓 Capstone Project · AI Crime Scene Analysis · "
    "YOLOv8 + FAISS RAG + Adam Optimizer + Claude Sonnet"
)