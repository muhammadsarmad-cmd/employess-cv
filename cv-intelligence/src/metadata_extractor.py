"""Extract structured metadata from CV text."""

import re
from typing import Dict, List, Any, Optional
from .config import COMMON_SKILLS


def extract_cv_metadata(text: str) -> Dict[str, Any]:
    """Extract structured metadata from CV text.

    Extracts:
    - Skills (matched against known skills list)
    - Years of experience (estimated)
    - Education level
    - Contact info (email, phone)
    """
    text_lower = text.lower()

    metadata = {
        "skills": extract_skills(text),
        "experience_years": extract_experience_years(text),
        "education_level": extract_education_level(text),
        "email": extract_email(text),
        "has_docker": "docker" in text_lower,
        "has_kubernetes": "kubernetes" in text_lower or "k8s" in text_lower,
        "has_ai_ml": any(term in text_lower for term in ["machine learning", "deep learning", "ai", "artificial intelligence", "ml"]),
        "has_cloud": any(term in text_lower for term in ["aws", "azure", "gcp", "cloud"]),
    }

    return metadata


def extract_skills(text: str) -> List[str]:
    """Extract skills from CV text."""
    text_lower = text.lower()
    found_skills = []

    for skill in COMMON_SKILLS:
        # Check for skill (case insensitive, word boundary)
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)

    return found_skills


def extract_experience_years(text: str) -> Optional[int]:
    """Extract total years of experience from CV."""
    # Common patterns for experience
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
        r'experience[:\s]*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*(?:in|as|of)',
    ]

    years_found = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            try:
                years = int(match)
                if 0 < years < 50:  # Reasonable range
                    years_found.append(years)
            except ValueError:
                continue

    if years_found:
        return max(years_found)  # Return highest mentioned

    # Fallback: count year ranges in work history
    year_pattern = r'(19|20)\d{2}\s*[-–]\s*(?:(19|20)\d{2}|present|current)'
    matches = re.findall(year_pattern, text.lower())
    if matches:
        # Rough estimate based on number of positions
        return min(len(matches) * 2, 20)

    return None


def extract_education_level(text: str) -> Optional[str]:
    """Extract highest education level."""
    text_lower = text.lower()

    # Check from highest to lowest
    if any(term in text_lower for term in ["ph.d", "phd", "doctorate", "doctor of"]):
        return "PhD"
    elif any(term in text_lower for term in ["master", "m.s.", "m.sc", "mba", "m.tech"]):
        return "Masters"
    elif any(term in text_lower for term in ["bachelor", "b.s.", "b.sc", "b.tech", "b.e.", "undergraduate"]):
        return "Bachelors"
    elif any(term in text_lower for term in ["associate", "diploma", "certification"]):
        return "Associate/Diploma"

    return None


def extract_email(text: str) -> Optional[str]:
    """Extract email address from CV."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(email_pattern, text)
    return match.group(0) if match else None


def extract_phone(text: str) -> Optional[str]:
    """Extract phone number from CV."""
    # Common phone patterns
    patterns = [
        r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)

    return None


def enrich_metadata_with_llm(text: str, llm) -> Dict[str, Any]:
    """Use LLM to extract more detailed metadata (optional, more accurate)."""
    prompt = f"""Extract the following information from this CV/Resume.
Return as JSON with these fields:
- name: candidate's full name
- skills: list of technical skills
- experience_years: total years of experience (number)
- current_title: current or most recent job title
- education: highest education with field
- summary: 2-sentence professional summary

CV Text:
{text[:3000]}

JSON:"""

    try:
        response = llm.invoke(prompt)
        import json
        return json.loads(response.content)
    except Exception as e:
        print(f"LLM metadata extraction failed: {e}")
        return {}
