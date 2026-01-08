"""Query engine for CV search and comparison."""

from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .config import GEMINI_API_KEY, LLM_MODEL
from .vector_store import CVVectorStore, load_cv_vectorstore


class CVQueryEngine:
    """Query engine for CV search, filtering, and comparison."""

    def __init__(self, vector_store: Optional[CVVectorStore] = None):
        self.vector_store = vector_store or load_cv_vectorstore()
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0
        )

    def search_candidates(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for candidates matching query and filters.

        Args:
            query: Natural language query (e.g., "Python developer with AWS experience")
            k: Number of results
            filters: Optional filters like {"experience_years_min": 5, "has_docker": True}
        """
        results = self.vector_store.search_with_scores(query, k=k, filter_dict=filters)

        candidates = []
        seen_files = set()

        for doc, score in results:
            source_file = doc.metadata.get("source_file", "unknown")

            # Deduplicate by source file
            if source_file in seen_files:
                continue
            seen_files.add(source_file)

            candidates.append({
                "source_file": source_file,
                "relevance_score": round(1 - score, 3),  # Convert distance to similarity
                "skills": doc.metadata.get("skills", []),
                "experience_years": doc.metadata.get("experience_years"),
                "education_level": doc.metadata.get("education_level"),
                "has_docker": doc.metadata.get("has_docker", False),
                "has_ai_ml": doc.metadata.get("has_ai_ml", False),
                "excerpt": doc.page_content[:300] + "..."
            })

        return candidates

    def find_best_candidates(
        self,
        job_requirement: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Find and rank best candidates for a job requirement.

        Example: "Docker deployment specialist with Kubernetes experience"
        """
        # Search with higher k to get more candidates
        candidates = self.search_candidates(job_requirement, k=k*2, filters=filters)

        if not candidates:
            return {"candidates": [], "summary": "No candidates found matching criteria."}

        # Use LLM to rank and explain
        candidates_text = "\n".join([
            f"Candidate {i+1} ({c['source_file']}): "
            f"Skills: {', '.join(c['skills'][:10])}, "
            f"Experience: {c['experience_years']} years, "
            f"Relevance: {c['relevance_score']}"
            for i, c in enumerate(candidates[:k])
        ])

        prompt = ChatPromptTemplate.from_template("""
You are a recruitment assistant. Based on the job requirement and candidate profiles,
provide a ranking and brief explanation for each candidate.

Job Requirement: {requirement}

Candidates:
{candidates}

Provide:
1. Ranked list (best to least fit) with brief justification
2. Key strengths of top candidate
3. Any gaps to consider

Response:""")

        response = self.llm.invoke(
            prompt.format(requirement=job_requirement, candidates=candidates_text)
        )

        return {
            "candidates": candidates[:k],
            "analysis": response.content
        }

    def compare_candidates(
        self,
        candidate_files: List[str],
        comparison_criteria: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare specific candidates.

        Args:
            candidate_files: List of CV file names to compare
            comparison_criteria: Optional specific criteria to focus on
        """
        # Get full content for each candidate
        candidates_data = []

        for filename in candidate_files:
            # Search for documents from this file
            results = self.vector_store.search(
                query=filename,
                k=20,
                filter_dict={"source_file": filename} if filename else None
            )

            # Find docs matching this file
            matching_docs = [d for d in results if filename in d.metadata.get("source_file", "")]

            if matching_docs:
                full_text = "\n".join(d.page_content for d in matching_docs[:5])
                metadata = matching_docs[0].metadata
                candidates_data.append({
                    "filename": filename,
                    "text": full_text[:2000],
                    "skills": metadata.get("skills", []),
                    "experience_years": metadata.get("experience_years"),
                    "education_level": metadata.get("education_level")
                })

        if len(candidates_data) < 2:
            return {"error": "Could not find enough candidates to compare"}

        # Format for LLM
        candidates_text = "\n\n".join([
            f"=== Candidate: {c['filename']} ===\n"
            f"Skills: {', '.join(c['skills'][:15])}\n"
            f"Experience: {c['experience_years']} years\n"
            f"Education: {c['education_level']}\n"
            f"Summary:\n{c['text'][:1000]}"
            for c in candidates_data
        ])

        criteria_text = f"Focus on: {comparison_criteria}" if comparison_criteria else "General comparison"

        prompt = ChatPromptTemplate.from_template("""
You are a recruitment assistant comparing candidates.

{criteria}

Candidates to compare:
{candidates}

Provide a detailed comparison:
1. Side-by-side comparison table (key attributes)
2. Strengths of each candidate
3. Weaknesses/gaps of each
4. Recommendation with reasoning

Response:""")

        response = self.llm.invoke(
            prompt.format(criteria=criteria_text, candidates=candidates_text)
        )

        return {
            "candidates": candidates_data,
            "comparison": response.content
        }

    def answer_query(self, query: str) -> Dict[str, Any]:
        """General query answering about CVs.

        Handles various query types:
        - "Who has Docker experience?"
        - "List candidates with 5+ years in AI"
        - "What skills does John have?"
        """
        # Use semantic search without strict filters
        # Filters can be too restrictive - let the LLM interpret results
        results = self.search_candidates(query, k=10, filters=None)

        if not results:
            return {"answer": "No candidates found matching your query.", "candidates": []}

        # Format context for LLM
        context = "\n".join([
            f"- {c['source_file']}: {', '.join(c['skills'][:8])}, "
            f"{c['experience_years']} years exp"
            for c in results[:10]
        ])

        prompt = ChatPromptTemplate.from_template("""
Based on the CV database, answer this recruitment query.

Query: {query}

Matching Candidates:
{context}

Provide a helpful answer with specific candidate recommendations.

Answer:""")

        response = self.llm.invoke(
            prompt.format(query=query, context=context)
        )

        return {
            "answer": response.content,
            "candidates": results[:5]
        }

    def _extract_filters_from_query(self, query: str) -> Dict[str, Any]:
        """Extract structured filters from natural language query."""
        query_lower = query.lower()
        filters = {}

        # Experience years
        import re
        exp_match = re.search(r'(\d+)\+?\s*years?', query_lower)
        if exp_match:
            years = int(exp_match.group(1))
            filters["experience_years_min"] = years

        # Technology filters
        if "docker" in query_lower:
            filters["has_docker"] = True
        if "kubernetes" in query_lower or "k8s" in query_lower:
            filters["has_kubernetes"] = True
        if any(term in query_lower for term in ["ai", "machine learning", "ml", "deep learning"]):
            filters["has_ai_ml"] = True
        if any(term in query_lower for term in ["aws", "azure", "gcp", "cloud"]):
            filters["has_cloud"] = True

        return filters


# Convenience function
def query_cvs(query: str) -> Dict[str, Any]:
    """Quick query interface."""
    engine = CVQueryEngine()
    return engine.answer_query(query)
