#!/usr/bin/env python3
"""Generate sample CV documents for testing."""

import random
from pathlib import Path
from datetime import datetime, timedelta

# Try to import docx, fall back to text files if not available
try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    print("python-docx not installed, generating .txt files instead")

SAMPLE_CVS_DIR = Path(__file__).parent / "data" / "sample_cvs"

# Sample data
FIRST_NAMES = ["John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa",
               "James", "Maria", "William", "Anna", "Richard", "Jennifer", "Thomas",
               "Amanda", "Daniel", "Jessica", "Christopher", "Ashley"]

LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
              "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas",
              "Moore", "Jackson", "Martin", "Lee", "Thompson", "White", "Harris"]

TITLES = [
    "Software Engineer", "Senior Software Engineer", "Staff Engineer",
    "DevOps Engineer", "Senior DevOps Engineer", "Platform Engineer",
    "Data Scientist", "Senior Data Scientist", "ML Engineer",
    "Backend Developer", "Frontend Developer", "Full Stack Developer",
    "Cloud Architect", "Solutions Architect", "Technical Lead",
    "Engineering Manager", "Product Engineer", "Site Reliability Engineer"
]

COMPANIES = [
    "Google", "Amazon", "Microsoft", "Meta", "Apple", "Netflix",
    "Uber", "Airbnb", "Stripe", "Shopify", "Salesforce", "Oracle",
    "IBM", "Intel", "Cisco", "VMware", "Adobe", "Dropbox",
    "Twitter", "LinkedIn", "Slack", "Zoom", "Atlassian", "GitHub"
]

SKILLS_POOL = {
    "languages": ["Python", "Java", "JavaScript", "TypeScript", "Go", "Rust", "C++", "C#", "Ruby", "Kotlin", "Swift"],
    "frameworks": ["React", "Angular", "Vue.js", "Django", "Flask", "FastAPI", "Spring Boot", "Node.js", "Express", ".NET"],
    "devops": ["Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins", "GitLab CI", "GitHub Actions", "ArgoCD"],
    "cloud": ["AWS", "Azure", "GCP", "Heroku", "DigitalOcean"],
    "databases": ["PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra", "DynamoDB"],
    "ai_ml": ["Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Scikit-learn", "NLP", "Computer Vision", "MLOps"],
    "other": ["Git", "Linux", "REST API", "GraphQL", "Microservices", "System Design", "Agile", "Scrum"]
}

UNIVERSITIES = [
    "MIT", "Stanford University", "Carnegie Mellon University",
    "UC Berkeley", "Georgia Tech", "University of Michigan",
    "Cornell University", "University of Washington", "UCLA",
    "University of Texas at Austin", "Purdue University", "Columbia University"
]

DEGREES = ["B.S. Computer Science", "B.S. Software Engineering", "B.S. Computer Engineering",
           "M.S. Computer Science", "M.S. Data Science", "M.S. Machine Learning",
           "MBA", "Ph.D. Computer Science"]


def generate_cv_content(profile_type: str = "random") -> dict:
    """Generate a random CV content."""
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    email = f"{first_name.lower()}.{last_name.lower()}@email.com"
    phone = f"+1-{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"

    # Experience based on profile type
    if profile_type == "senior":
        exp_years = random.randint(8, 15)
        title = random.choice(["Senior Software Engineer", "Staff Engineer", "Technical Lead", "Engineering Manager"])
    elif profile_type == "devops":
        exp_years = random.randint(3, 10)
        title = random.choice(["DevOps Engineer", "Senior DevOps Engineer", "Platform Engineer", "SRE"])
    elif profile_type == "ai":
        exp_years = random.randint(3, 10)
        title = random.choice(["Data Scientist", "ML Engineer", "Senior Data Scientist", "AI Engineer"])
    else:
        exp_years = random.randint(2, 12)
        title = random.choice(TITLES)

    # Skills based on profile
    skills = []
    skills.extend(random.sample(SKILLS_POOL["languages"], random.randint(2, 4)))
    skills.extend(random.sample(SKILLS_POOL["frameworks"], random.randint(1, 3)))
    skills.extend(random.sample(SKILLS_POOL["databases"], random.randint(1, 3)))
    skills.extend(random.sample(SKILLS_POOL["other"], random.randint(2, 4)))

    if profile_type == "devops" or random.random() > 0.5:
        skills.extend(random.sample(SKILLS_POOL["devops"], random.randint(2, 4)))
        skills.extend(random.sample(SKILLS_POOL["cloud"], random.randint(1, 2)))

    if profile_type == "ai" or random.random() > 0.6:
        skills.extend(random.sample(SKILLS_POOL["ai_ml"], random.randint(2, 4)))

    # Generate work history
    jobs = []
    current_year = datetime.now().year
    year = current_year
    remaining_years = exp_years

    while remaining_years > 0:
        job_years = min(random.randint(1, 4), remaining_years)
        jobs.append({
            "title": random.choice(TITLES),
            "company": random.choice(COMPANIES),
            "start": year - job_years,
            "end": year if year < current_year else "Present",
            "years": job_years
        })
        year -= job_years
        remaining_years -= job_years

    # Education
    degree = random.choice(DEGREES)
    university = random.choice(UNIVERSITIES)
    grad_year = current_year - exp_years - random.randint(0, 2)

    return {
        "name": f"{first_name} {last_name}",
        "email": email,
        "phone": phone,
        "title": title,
        "experience_years": exp_years,
        "skills": list(set(skills)),
        "jobs": jobs,
        "education": {"degree": degree, "university": university, "year": grad_year}
    }


def format_cv_text(cv: dict) -> str:
    """Format CV as plain text."""
    text = f"""
{cv['name']}
{cv['title']}

Contact:
Email: {cv['email']}
Phone: {cv['phone']}

PROFESSIONAL SUMMARY
Experienced {cv['title']} with {cv['experience_years']}+ years of experience in software development.
Skilled in {', '.join(cv['skills'][:5])} and other technologies.

SKILLS
{', '.join(cv['skills'])}

EXPERIENCE
"""
    for job in cv['jobs']:
        text += f"""
{job['title']} at {job['company']}
{job['start']} - {job['end']}
- Developed and maintained software applications
- Collaborated with cross-functional teams
- Implemented best practices and coding standards
"""

    text += f"""
EDUCATION
{cv['education']['degree']}
{cv['education']['university']}
Graduated: {cv['education']['year']}
"""
    return text


def generate_sample_cvs(num_samples: int = 30):
    """Generate sample CV files."""
    SAMPLE_CVS_DIR.mkdir(parents=True, exist_ok=True)

    # Distribution of profile types
    profiles = (
        ["senior"] * 5 +
        ["devops"] * 8 +
        ["ai"] * 7 +
        ["random"] * (num_samples - 20)
    )
    random.shuffle(profiles)

    generated = []

    for i, profile_type in enumerate(profiles[:num_samples]):
        cv = generate_cv_content(profile_type)
        cv_text = format_cv_text(cv)

        # Create filename
        name_slug = cv['name'].lower().replace(' ', '_')
        filename = f"{name_slug}_cv.txt"
        filepath = SAMPLE_CVS_DIR / filename

        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cv_text)

        generated.append({
            "filename": filename,
            "name": cv['name'],
            "profile": profile_type,
            "experience": cv['experience_years'],
            "has_docker": "Docker" in cv['skills'],
            "has_ai": any(s in cv['skills'] for s in SKILLS_POOL['ai_ml'])
        })

        print(f"Generated: {filename} ({profile_type}, {cv['experience_years']} years)")

    print(f"\n[OK] Generated {len(generated)} sample CVs in {SAMPLE_CVS_DIR}")

    # Summary
    docker_count = sum(1 for g in generated if g['has_docker'])
    ai_count = sum(1 for g in generated if g['has_ai'])
    senior_count = sum(1 for g in generated if g['experience'] >= 5)

    print(f"\nSummary:")
    print(f"  - With Docker experience: {docker_count}")
    print(f"  - With AI/ML experience: {ai_count}")
    print(f"  - With 5+ years experience: {senior_count}")

    return generated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate sample CVs for testing")
    parser.add_argument("-n", "--num", type=int, default=30, help="Number of CVs to generate")
    args = parser.parse_args()

    generate_sample_cvs(args.num)
