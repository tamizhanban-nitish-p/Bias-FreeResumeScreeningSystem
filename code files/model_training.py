from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def simulate_target(feature_matrix):
    """Simulate target labels for resumes."""
    return np.random.randint(0, 2, size=feature_matrix.shape[0])  # 0 or 1 for each resume

if __name__ == "__main__":
    from feature_extraction import create_feature_matrix_from_file

    resumes_file = "processed_resumes.txt"
    skillset = [
        "java", "selenium", "appium", "jira", "soapui", "postman", "agile", "sql",
        "javascript", "python", "c++", "html", "css", "maven", "jenkins", "rest",
        "api", "webdriver", "testng", "linux", "windows"
    ]

    # Generate feature matrix and simulate target labels
    feature_matrix, _ = create_feature_matrix_from_file(resumes_file, skillset)
    target = simulate_target(feature_matrix)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, target, test_size=0.3, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
