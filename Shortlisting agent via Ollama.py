import os
import fitz  # PyMuPDF — reads PDF files without external tools
import csv
import datetime
import requests
import time
import pyttsx3

# ---------------------------
# OLLAMA SETTINGS
# ---------------------------
OLLAMA_MODEL = "qwen2.5:7b"                       # model served by the local Ollama daemon
OLLAMA_URL = "http://localhost:11434/api/generate"  # default Ollama REST endpoint

# ---------------------------
# TEXT-TO-SPEECH
# ---------------------------
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()

# ---------------------------
# EXTRACT TEXT FROM PDF
# ---------------------------
def extract_text_from_pdf(file_path):
    """Read all pages of a PDF and return a single concatenated string."""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""  # return empty string so downstream code can still run

# ---------------------------
# SCORE CV AGAINST JOB DESCRIPTION (OLLAMA)
# ---------------------------
def score_cv_with_ollama(cv_text, job_description, retries=3):
    """
    Send a CV + job description to the local Ollama model and get back a
    numeric match score (0–100) plus a short explanation.
    Retries up to `retries` times with exponential back-off on failure.
    """
    prompt = f"""
Compare the following candidate CV to the job description and provide:
1. A match score (0 to 100)
2. A brief explanation (40–50 words) explaining why the candidate was or was not a good fit.

Job Description:
{job_description}

Candidate CV:
{cv_text}

Output format:
Score: <number>
Reason: <40–50 word explanation>
"""
    for attempt in range(retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False   # wait for the full response before returning
                }
            )
            response.raise_for_status()
            return response.json().get("response", "No response.")
        except Exception as e:
            print(f"[!] Error on attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                # Back-off: 10 s, 20 s, 30 s … to avoid hammering a busy model server
                time.sleep(10 + attempt * 10)
            else:
                return f"Error scoring CV with Ollama: {e}"

# ---------------------------
# PROCESS A SINGLE CV FILE
# ---------------------------
def process_single_cv(file_path, job_description):
    """Extract text from one PDF and return a (filename, raw_llm_response) tuple."""
    if not file_path.lower().endswith(".pdf"):
        print(f"Skipped non-PDF file: {file_path}")
        return []

    print(f"Processing {file_path}...")
    cv_text = extract_text_from_pdf(file_path)
    # Truncate to 12 000 chars to stay within the model's context window
    result = score_cv_with_ollama(cv_text[:12000], job_description)
    return [(os.path.basename(file_path), result)]

# ---------------------------
# PROCESS ALL CVS IN A FOLDER
# ---------------------------
def process_all_cvs(folder_path, job_description):
    """Iterate over every PDF in a folder and collect LLM responses."""
    all_results = []
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if file_path.lower().endswith(".pdf"):
            results = process_single_cv(file_path, job_description)
            all_results.extend(results)
    return all_results

# ---------------------------
# PARSE RESULTS AND SHORTLIST
# ---------------------------
def parse_and_shortlist(results, threshold=70):
    """
    Parse raw LLM text for 'Score:' and 'Reason:' lines.
    Candidates with score >= threshold are marked as shortlisted.
    """
    all_data = []
    for file, result in results:
        try:
            lines = result.splitlines()
            # Find the first line that contains the score label
            score_line = next((line for line in lines if "Score:" in line), "")
            reason_line = next((line for line in lines if "Reason:" in line), "")
            score = int(score_line.split(":", 1)[1].strip()) if score_line else 0
            reason = reason_line.split(":", 1)[1].strip() if reason_line else "No reason provided."
            shortlisted = "Yes" if score >= threshold else "No"
            all_data.append({
                "Filename": file,
                "Score": score,
                "Shortlisted": shortlisted,
                "Reason": reason
            })
        except Exception as e:
            print(f"Error parsing result for {file}: {e}")
            # Keep a row even on failure so no CV silently disappears from the output
            all_data.append({
                "Filename": file,
                "Score": 0,
                "Shortlisted": "No",
                "Reason": f"Failed to parse score: {e}"
            })
    return all_data

# ---------------------------
# WRITE TO CSV
# ---------------------------
def export_to_csv(data, output_folder=r"C:\Users\JBS\Downloads\results"):
    """Write parsed results to a timestamped CSV file in the output folder."""
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_folder, f"cv_results_{timestamp}.csv")
    print(f"[i] Attempting to write CSV to: {output_path}")
    keys = ["Filename", "Score", "Shortlisted", "Reason"]
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
        print(f"\n[✓] Results saved to: {output_path}")
    except Exception as e:
        print(f"[!] Failed to write CSV: {e}")

# ---------------------------
# MAIN PROGRAM
# ---------------------------
if __name__ == "__main__":
    # Define the target role — change this block to screen for a different position
    job_description = """
We are looking for a Software Engineer with 2+ years of experience in Python and web development.
Strong understanding of REST APIs, databases (PostgreSQL), and cloud services (AWS/GCP) is required.
Experience with React.js is a plus.
"""
    # Folder containing candidate PDF resumes
    folder_path = "C:/Users/JBS/Downloads/dummy_cvs"

    speak("Starting CV screening. Please wait.")
    results = process_all_cvs(folder_path, job_description)

    # Print raw LLM responses for debugging before parsing
    print("\n--- RAW AI RESPONSES ---")
    for file, result in results:
        print(f"\n{file}:\n{result}")

    parsed_data = parse_and_shortlist(results)
    shortlisted = [d for d in parsed_data if d["Shortlisted"] == "Yes"]

    print("\n--- SHORTLISTED CANDIDATES ---")
    for d in shortlisted:
        print(f"{d['Filename']} - Score: {d['Score']} - Reason: {d['Reason']}")

    export_to_csv(parsed_data)
    speak(f"Screening complete. {len(shortlisted)} candidate{'s' if len(shortlisted) != 1 else ''} shortlisted out of {len(parsed_data)}.")
