"""Interactive Streamlit interface for grounded FiQA question answering."""

import streamlit as st

from utils.qa import answer_question
from utils.retrievers import BM25Retriever, DenseRetriever, load_corpus


@st.cache_resource
def load_retrievers():
    return (
        BM25Retriever(),
        DenseRetriever(),
        load_corpus().set_index("_id"),
    )


st.set_page_config(page_title="FiQA Q&A", page_icon="🔎")
st.title("FiQA question answering")
st.write(
    "Ask a question about the FiQA financial Q&A dataset. "
    "The answer is generated from retrieved documents and includes source IDs."
)

with st.form("question_form"):
    question = st.text_area(
        "Your question",
        placeholder="Where should I keep my emergency fund?",
    )
    top_k = st.slider("Number of source documents", min_value=1, max_value=10, value=5)
    submitted = st.form_submit_button("Get answer")

if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Searching FiQA and generating an answer..."):
            bm25, dense, corpus_by_id = load_retrievers()
            answer, sources = answer_question(
                question, bm25, dense, corpus_by_id, top_k=top_k
            )

        st.subheader("Answer")
        st.write(answer)

        if sources:
            st.subheader("Retrieved sources")
            for doc_id, text, score in sources:
                with st.expander(f"Source {doc_id} · RRF score {score:.4f}"):
                    st.write(text)
