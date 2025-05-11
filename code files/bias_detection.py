import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Simulate demographic data
def simulate_demographics(num_resumes):
    genders = np.random.choice(["Male", "Female", "Non-Binary"], size=num_resumes)
    return genders

if __name__ == "__main__":
    from feature_extraction import create_feature_matrix_from_file

    resumes_file = "processed_resumes.txt"
    skillset = ["python", "java", "machine learning", "data analysis", "nlp", "deep learning", "frontend development"]

    # Generate feature matrix and simulate target
    feature_matrix, _ = create_feature_matrix_from_file(resumes_file, skillset)
    target = np.random.randint(0, 2, size=feature_matrix.shape[0])  # Simulated target
    sensitive_features = simulate_demographics(feature_matrix.shape[0])

    # Train-test split
    X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
        feature_matrix, target, sensitive_features, test_size=0.3, random_state=42
    )

    # Train a Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate bias
    predictions = model.predict(X_test)
    metrics = MetricFrame(
        metrics=selection_rate,
        y_true=y_test,
        y_pred=predictions,
        sensitive_features=sf_test
    )
    print("Selection Rates by Group:")
    print(metrics.by_group)
