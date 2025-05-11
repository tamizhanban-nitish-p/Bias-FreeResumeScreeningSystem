from sklearn.feature_extraction.text import CountVectorizer

def create_feature_matrix_from_file(file_path, skillset):
    """Create a feature matrix from processed resumes stored in a text file."""
    with open(file_path, "r") as f:
        resumes = f.read().split("\n\n")  # Split resumes using a double newline
        resumes = [resume.split("Text:")[1].strip() for resume in resumes if "Text:" in resume]

    # Convert resumes to a feature matrix
    vectorizer = CountVectorizer(vocabulary=skillset)
    feature_matrix = vectorizer.fit_transform(resumes).toarray()
    return feature_matrix, resumes

if __name__ == "__main__":
    resumes_file = "processed_resumes.txt"
    skillset = [
        "java", "selenium", "appium", "jira", "soapui", "postman", "agile", "sql",
        "javascript", "python", "c++", "html", "css", "maven", "jenkins", "rest",
        "api", "webdriver", "testng", "linux", "windows"
    ]

    # Generate feature matrix
    feature_matrix, resumes_list = create_feature_matrix_from_file(resumes_file, skillset)

    print("Feature Matrix:")
    print(feature_matrix)

    print("\nResumes List:")
    for i, resume in enumerate(resumes_list):
        print(f"Resume {i + 1}: {resume}")
