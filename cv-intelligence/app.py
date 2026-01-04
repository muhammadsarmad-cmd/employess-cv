"""Streamlit UI for CV Intelligence System."""

import streamlit as st
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.query_engine import CVQueryEngine
from src.vector_store import CVVectorStore, load_cv_vectorstore
from src.document_processor import CVDocumentProcessor
from src.config import SAMPLE_CVS_DIR, DATA_DIR


def init_session_state():
    """Initialize session state."""
    if "query_engine" not in st.session_state:
        try:
            st.session_state.query_engine = CVQueryEngine()
            st.session_state.db_loaded = True
        except Exception as e:
            st.session_state.db_loaded = False
            st.session_state.error = str(e)


def main():
    st.set_page_config(
        page_title="CV Intelligence System",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 CV Intelligence System")
    st.markdown("*AI-powered recruitment assistant*")

    init_session_state()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Database status
        if st.session_state.get("db_loaded"):
            st.success("✅ Database connected")
            try:
                store = load_cv_vectorstore()
                info = store.get_collection_info()
                st.metric("Documents indexed", info.get("points_count", "N/A"))
            except:
                pass
        else:
            st.warning("⚠️ Database not loaded")
            st.info("Run `python ingest.py` to index CVs")

        st.divider()

        # Ingestion section
        st.header("📥 Ingest CVs")
        cv_dir = st.text_input(
            "CV Directory",
            value=str(SAMPLE_CVS_DIR),
            help="Path to folder containing CVs"
        )

        if st.button("🔄 Index CVs", type="primary"):
            with st.spinner("Processing CVs..."):
                try:
                    processor = CVDocumentProcessor()
                    chunks = processor.process_directory(cv_dir)

                    if chunks:
                        store = CVVectorStore()
                        store.create_collection(chunks)
                        st.session_state.query_engine = CVQueryEngine(store)
                        st.session_state.db_loaded = True
                        st.success(f"✅ Indexed {len(chunks)} chunks!")
                        st.rerun()
                    else:
                        st.error("No documents found")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Main content - Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "🏆 Find Best", "⚖️ Compare"])

    # Tab 1: General Search
    with tab1:
        st.header("Search Candidates")

        query = st.text_input(
            "Enter your query",
            placeholder="e.g., Python developer with AWS experience",
            key="search_query"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            min_exp = st.number_input("Min Experience (years)", 0, 30, 0)
        with col2:
            require_docker = st.checkbox("Requires Docker")
        with col3:
            require_ai = st.checkbox("Requires AI/ML")

        if st.button("🔍 Search", key="search_btn") and query:
            if not st.session_state.get("db_loaded"):
                st.error("Please index CVs first!")
            else:
                with st.spinner("Searching..."):
                    filters = {}
                    if min_exp > 0:
                        filters["experience_years_min"] = min_exp
                    if require_docker:
                        filters["has_docker"] = True
                    if require_ai:
                        filters["has_ai_ml"] = True

                    engine = st.session_state.query_engine
                    result = engine.answer_query(query)

                    st.subheader("Answer")
                    st.write(result["answer"])

                    st.subheader("Matching Candidates")
                    for i, c in enumerate(result.get("candidates", [])[:5]):
                        with st.expander(f"📄 {c['source_file']} (Score: {c['relevance_score']})"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Experience:** {c.get('experience_years', 'N/A')} years")
                                st.write(f"**Education:** {c.get('education_level', 'N/A')}")
                            with col2:
                                st.write(f"**Docker:** {'✅' if c.get('has_docker') else '❌'}")
                                st.write(f"**AI/ML:** {'✅' if c.get('has_ai_ml') else '❌'}")
                            st.write(f"**Skills:** {', '.join(c.get('skills', [])[:10])}")
                            st.write(f"**Excerpt:** {c.get('excerpt', '')}")

    # Tab 2: Find Best Candidates
    with tab2:
        st.header("Find Best Candidates for Role")

        job_req = st.text_area(
            "Job Requirements",
            placeholder="e.g., Senior DevOps Engineer with Docker and Kubernetes expertise, 5+ years experience",
            height=100,
            key="job_req"
        )

        num_candidates = st.slider("Number of candidates", 3, 10, 5)

        if st.button("🏆 Find Best", key="best_btn") and job_req:
            if not st.session_state.get("db_loaded"):
                st.error("Please index CVs first!")
            else:
                with st.spinner("Analyzing candidates..."):
                    engine = st.session_state.query_engine
                    result = engine.find_best_candidates(job_req, k=num_candidates)

                    st.subheader("Analysis")
                    st.write(result["analysis"])

                    st.subheader("Top Candidates")
                    for i, c in enumerate(result.get("candidates", [])):
                        st.write(f"**{i+1}. {c['source_file']}** - Score: {c['relevance_score']}")

    # Tab 3: Compare Candidates
    with tab3:
        st.header("Compare Candidates")

        st.write("Enter CV filenames to compare (comma-separated):")
        candidates_input = st.text_input(
            "Candidate files",
            placeholder="e.g., john_doe.pdf, jane_smith.pdf",
            key="compare_input"
        )

        comparison_focus = st.text_input(
            "Comparison focus (optional)",
            placeholder="e.g., backend development skills, cloud experience",
            key="compare_focus"
        )

        if st.button("⚖️ Compare", key="compare_btn") and candidates_input:
            if not st.session_state.get("db_loaded"):
                st.error("Please index CVs first!")
            else:
                candidates = [c.strip() for c in candidates_input.split(",")]

                if len(candidates) < 2:
                    st.warning("Please enter at least 2 candidates to compare")
                else:
                    with st.spinner("Comparing candidates..."):
                        engine = st.session_state.query_engine
                        result = engine.compare_candidates(
                            candidates,
                            comparison_focus if comparison_focus else None
                        )

                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.subheader("Comparison")
                            st.write(result["comparison"])


if __name__ == "__main__":
    main()
