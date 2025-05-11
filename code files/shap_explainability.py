import shap
import numpy as np  # Import NumPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    from feature_extraction import create_feature_matrix_from_file

    resumes_file = "processed_resumes.txt"
    skillset = [
        "java", "selenium", "appium", "jira", "soapui", "postman", "agile", "sql",
        "javascript", "python", "c++", "html", "css", "maven", "jenkins", "rest",
        "api", "webdriver", "testng", "linux", "windows"
    ]

    # Generate feature matrix and simulate target
    feature_matrix, _ = create_feature_matrix_from_file(resumes_file, skillset)
    target = np.random.randint(0, 2, size=feature_matrix.shape[0])  # Simulated target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, target, test_size=0.3, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP summary plot
    shap.summary_plot(shap_values, X_test, feature_names=skillset)
