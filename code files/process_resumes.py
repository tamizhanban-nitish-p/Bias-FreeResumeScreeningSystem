import os
from docx import Document
import re

def extract_text_from_docx(file_path):
    """Extract text from a Word document (.docx)."""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def clean_resume_text(text):
    """Clean and preprocess the resume text."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text.strip()

def process_resumes(folder_path):
    """Process all resumes in a folder and return cleaned text."""
    resumes_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):  # Only process .docx files
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {file_path}")
            raw_text = extract_text_from_docx(file_path)
            cleaned_text = clean_resume_text(raw_text)
            resumes_data.append({"filename": filename, "text": cleaned_text})
    return resumes_data

if __name__ == "__main__":
    resumes_folder = "./Resumes"  # Path to your Resumes folder
    resumes = process_resumes(resumes_folder)
    
    # Save the processed resumes to a file for further use
    with open("processed_resumes.txt", "w", encoding="utf-8") as f:  # Specify UTF-8 encoding
        for resume in resumes:
            f.write(f"Filename: {resume['filename']}\n")
            f.write(f"Text: {resume['text']}\n\n")
