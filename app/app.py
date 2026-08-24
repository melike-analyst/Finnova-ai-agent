"""
FinNova Bank - Otonom İş Zekası Ajanı - Streamlit arayüzü
Çalıştırmak için: streamlit run app/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.agent import ask  # noqa: E402

st.set_page_config(page_title="FinNova BI Agent", layout="centered")
st.title(" FinNova Bank — Otonom İş Zekası Ajanı")
st.caption(
    "Doğal dilde bir soru sorun; agent veritabanını sorgular, analiz eder "
    "ve gerekiyorsa grafik üretir. (Veri tamamen sentetiktir.)"
)

EXAMPLE_QUESTIONS = [
    "Hangi işlem tipinde şüpheli işlem oranı en yüksek?",
    "Son 6 ayda İstanbul ve İzmir'de işlem hacmi nasıl değişti?",
    "Kış aylarında fatura ödemesi oranı diğer mevsimlere göre nasıl?",
    "Kredi skoru 550'nin altındaki müşterilerin ortalama hesap bakiyesi nedir?",
]

with st.sidebar:
    st.subheader("Örnek sorular")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state["pending_question"] = q

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("chart_path"):
            st.image(turn["chart_path"])

question = st.chat_input("Bir soru sorun...") or st.session_state.pop("pending_question", None)

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analiz ediliyor..."):
            result = ask(question)
        st.markdown(result["answer"])
        if result.get("chart_path"):
            st.image(result["chart_path"])

    st.session_state.history.append({
        "role": "assistant",
        "content": result["answer"],
        "chart_path": result.get("chart_path"),
    })
