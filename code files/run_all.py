import os
import subprocess

def run_script(script_name):
    """Run a Python script and print its output."""
    try:
        print(f"\nRunning {script_name}...\n{'-'*50}")
        result = subprocess.run(["python", script_name], check=True, capture_output=True, text=True)
        print(result.stdout)  # Print the output of the script
        print(f"\n{script_name} completed successfully.\n{'-'*50}")
    except subprocess.CalledProcessError as e:
        print(f"\nError while running {script_name}:\n{e.stderr}\n{'-'*50}")

if __name__ == "__main__":
    # List of scripts to run in sequence
    scripts = [
        "process_resumes.py",
        "feature_extraction.py",
        "model_training.py",
        "bias_detection.py",
        "shap_explainability.py"
    ]
    
    # Ensure the working directory is correct
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    # Run each script
    for script in scripts:
        run_script(script)
