import re
from pdfminer.high_level import extract_text
import spacy
from spacy.matcher import Matcher

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def extract_contact_info(text):
    contact_info = {"name": None, "phone": None, "email": None, "address": None}

    # Extract Name using Spacy
    contact_info["name"] = extract_name(text)

    # Extract Phone Number
    phone_pattern = r"\b(?:\+1\s?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    phone_match = re.search(phone_pattern, text)
    contact_info["phone"] = phone_match.group() if phone_match else None

    # Extract Email
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    email_match = re.search(email_pattern, text)
    contact_info["email"] = email_match.group() if email_match else None

    # Extract Address (Assumes address format with city, state, and ZIP)
    address_pattern = r"Address:\s*([\w\s,]+)\s*\|\s*Phone"
    address_match = re.search(address_pattern, text)
    contact_info["address"] = address_match.group(1) if address_match else None

    return contact_info

def extract_name(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def extract_skills(text, skills_list):
    skills_found = []
    for skill in skills_list:
        pattern = rf"\b{re.escape(skill)}\b"
        if re.search(pattern, text, re.IGNORECASE):
            skills_found.append(skill)
    return skills_found

def extract_work_experience(text):
    work_experience = []
    
    work_pattern = re.compile(
        r"(?P<job_title>[\w\s\-]+)\s*–\s*(?P<company>[\w\s&.,]+)\s*\|\s*(?P<location>[\w\s,]+)?\s*\|\s*(?P<start_date>\w+\s\d{4})\s*–\s*(?P<end_date>\w+\s\d{4}|Present)",
        re.IGNORECASE
    )

    matches = work_pattern.finditer(text)
    for match in matches:
        work_experience.append({
            "Job Title": match.group("job_title"),
            "Company": match.group("company"),
            "Location": match.group("location") if match.group("location") else None,
            "Start Date": match.group("start_date"),
            "End Date": match.group("end_date"),
        })
    
    return work_experience

def extract_research(text):
    research_experience = []
    
    research_pattern = re.compile(
        r"(?P<title>[\w\s]+)\s*–\s*(?P<organization>[\w\s]+)\s*\|\s*(?P<location>[\w\s,]+)?\s*\|\s*(?P<start_date>\w+\s\d{4})\s*–\s*(?P<end_date>\w+\s\d{4}|Present)",
        re.IGNORECASE
    )

    matches = research_pattern.finditer(text)
    for match in matches:
        research_experience.append({
            "Title": match.group("title"),
            "Organization": match.group("organization"),
            "Location": match.group("location") if match.group("location") else None,
            "Start Date": match.group("start_date"),
            "End Date": match.group("end_date"),
        })
    
    return research_experience

def extract_projects(text):
    projects = []
    
    project_pattern = re.compile(
        r"(?P<title>[\w\s]+)\s*\|\s*(?P<location>[\w\s,]+)?\s*\|\s*(?P<start_date>\w+\s\d{4})\s*–\s*(?P<end_date>\w+\s\d{4}|Present)",
        re.IGNORECASE
    )

    matches = project_pattern.finditer(text)
    for match in matches:
        projects.append({
            "Title": match.group("title"),
            "Location": match.group("location") if match.group("location") else None,
            "Start Date": match.group("start_date"),
            "End Date": match.group("end_date"),
        })
    
    return projects

if __name__ == '__main__':
    resume_path = "/Users/aryavratgupta/Desktop/Aryavrat_Gupta_Resume_Full-Stack_AI_ML.pdf"
    text = extract_text_from_pdf(resume_path)

    print("\n--- Contact Information ---")
    contact_info = extract_contact_info(text)
    for key, value in contact_info.items():
        print(f"{key.capitalize()}: {value}")

    print("\n--- Skills ---")
    skills_list = ['Python', 'Data Analysis', 'Machine Learning', 'Java', 'React.js', 'Docker', 'AWS', 'SQL', 'Spring Boot', 'TensorFlow', 'PyTorch']
    skills_found = extract_skills(text, skills_list)
    print("Skills:", skills_found)

    print("\n--- Work Experience ---")
    work_experience = extract_work_experience(text)
    for work in work_experience:
        print(work)

    print("\n--- Research Experience (Optional) ---")
    research_experience = extract_research(text)
    for research in research_experience:
        print(research)

    print("\n--- Projects (Optional) ---")
    projects = extract_projects(text)
    for project in projects:
        print(project)
