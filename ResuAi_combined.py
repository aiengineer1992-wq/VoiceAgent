import streamlit as st
import pandas as pd
import sounddevice as sd
from faster_whisper import WhisperModel
import pyttsx3
import ollama
import os
import re
import fitz  # PyMuPDF
import docx
import requests
import time
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor, as_completed
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(page_title="ResuAI End-to-End", layout="wide")
# -------------------------------
# 🌐 Configuration
# -------------------------------
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"
SHORTLIST_THRESHOLD_DEFAULT = 70
RESUME_PREVIEW_CHARS = 12000

# -------------------------------
# 📄 Text Extraction
# -------------------------------
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "".join(page.get_text() for page in doc)

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs)

def extract_text(file_path):
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    if file_path.lower().endswith(".docx"):
        return extract_text_from_docx(file_path)
    return ""

# -------------------------------
# 📞 Contact Extraction
# -------------------------------
def extract_contact_info(text):
    name = re.search(r"(?i)name[:\s]*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", text)
    email = re.search(r"[\w\.-]+@[\w\.-]+", text)
    phone = re.search(r"\+?\d[\d\s\-\(\)]{7,}\d", text)
    return (
        name.group(1) if name else "",
        phone.group(0) if phone else "",
        email.group(0) if email else ""
    )

# -------------------------------
# 🤖 Shortlisting via Ollama
# -------------------------------
def score_cv_with_ollama(cv_text, job_description, retries=3):
    prompt = f"""
You are an AI recruiter evaluating a candidate's resume against a job description.
Provide a match score 0-100 and a 40-50 word reason.
Job Description: {job_description}
Candidate CV: {cv_text}
Output format: Score: <number>\nReason: <text>
"""
    for i in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False})
            r.raise_for_status()
            return r.json().get("response", "Score:0\nReason:No response.")
        except:
            time.sleep(2 * (i+1))
    return "Score:0\nReason:Error calling model."

# -------------------------------
# 📝 Parse & Shortlist
# -------------------------------
def parse_and_shortlist(results, threshold):
    all_data, contacts = [], []
    for fname, res, tmp_path in results:
        score = int(re.search(r"Score:\s*(\d+)", res).group(1)) if re.search(r"Score:\s*(\d+)", res) else 0
        reason = re.search(r"Reason:\s*(.+)", res, re.DOTALL).group(1).strip() if re.search(r"Reason:\s*(.+)", res, re.DOTALL) else "No reason provided."
        shortlisted = score >= threshold
        all_data.append({
            "Filename": fname,
            "Score": score,
            "Shortlisted": "Yes" if shortlisted else "No",
            "Reason": reason
        })
        if shortlisted:
            txt = extract_text(tmp_path)
            name, phone, email = extract_contact_info(txt)
            # Use extracted name or filename fallback
            contacts.append({"Name": name or fname, "Phone": phone, "Email": email})
    return all_data, contacts

# -------------------------------
# 🔉 Voice Interview Models
# -------------------------------
@st.cache_resource
def load_voice_models():
    return WhisperModel("tiny", compute_type="int8")

whisper_model = load_voice_models()

def play_audio(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def record_audio(duration=10, fs=16000):
    st.info("🎙 Recording...")
    audio = sd.rec(int(duration*fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    st.success("Recording complete.")
    return audio[:,0]

def transcribe_audio(audio):
    segments, _ = whisper_model.transcribe(audio, language="en")
    return " ".join(seg.text for seg in segments)

# -------------------------------
# 🎯 Interview Scoring & Sentiment
# -------------------------------
def validate_answer(question, answer):
    prompt = f"""You are an interview assistant. Determine if the answer is valid for the given question.

Question: "{question}"
Answer: "{answer}"

Reply with only "Valid Answer" if the answer is appropriate, otherwise reply with "Invalid Answer"."""
    
    try:
        response = ollama.chat(
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content'].strip()
    except Exception as e:
        return "No"


def score_interview(answers):
    qas = "\n".join(f"Q: {q}\nA: {a}" for q, a in answers.items())
    prompt = f"""
You are a professional HR expert conducting voice interviews. 
Evaluate the candidate's answers critically, but with respect.

Your goal is to help the company hire top talent by scoring the interview between 0 and 100 based on:
- Relevance and clarity of answers
- Communication quality
- Role fit

Avoid harsh language. Be honest, constructive, and insightful.
Output format:
Score: <number>
Reason: <brief but clear explanation of strengths and concerns>
---
{qas}
"""
    out = ollama.chat(model="qwen2.5:7b", messages=[{"role": "user", "content": prompt}])
    txt = out["message"]["content"]
    m = re.search(r"(\d{1,3})", txt)
    return (int(m.group()), txt[m.end():].strip()) if m else (0, txt)


def analyze_sentiment(all_text):
    txt = " ".join(all_text)
    prompt = f"Sentiment? Positive/Neutral/Negative.\n{txt}"
    out = ollama.chat(model="qwen2.5:7b", messages=[{"role":"user","content":prompt}])
    s = re.search(r"(Positive|Neutral|Negative)", out["message"]["content"], re.I)
    return (s.group().capitalize(), out["message"]["content"]) if s else ("Unknown", out["message"]["content"])

# -------------------------------
# 🚀 Main Application
# -------------------------------
def main():
    # NOTE: st.set_page_config already called above

    st.title("ResuAI: Shortlist & Voice Interview")

    # 1️⃣ Shortlisting
    st.header("1. Shortlisting")
    jd = st.text_area("Job Description", height=150)
    files = st.file_uploader("Upload CVs (PDF/DOCX)", type=["pdf","docx"], accept_multiple_files=True)
    thresh = st.slider("Score threshold", 0, 100, SHORTLIST_THRESHOLD_DEFAULT)

    if st.button("Run Shortlist"):
        if not jd or not files:
            st.error("Provide JD and at least one CV.")
            return
        with TemporaryDirectory() as tmp:
            results = []
            progress_bar = st.progress(0)
            for i, f in enumerate(files):
                fp = os.path.join(tmp, f.name)
                open(fp, "wb").write(f.read())
                res = score_cv_with_ollama(extract_text(fp)[:RESUME_PREVIEW_CHARS], jd)
                results.append((f.name, res, fp))
                progress_bar.progress((i + 1) / len(files))
            progress_bar.empty()

            all_data, contacts = parse_and_shortlist(results, thresh)

            # Full results CSV
            df_all = pd.DataFrame(all_data)
            st.subheader("All CV Results")
            st.dataframe(df_all)
            st.download_button("📥 Download All Results (CSV)", df_all.to_csv(index=False).encode(), "cv_results.csv", mime="text/csv")

            # Shortlisted contacts CSV
            df_contacts = pd.DataFrame(contacts)
            st.subheader("Shortlisted Contacts")
            st.dataframe(df_contacts)
            st.download_button("📥 Download Shortlisted Contacts (CSV)", df_contacts.to_csv(index=False).encode(), "shortlisted_contacts.csv", mime="text/csv")

            # Initialize interview
            st.session_state.candidates = contacts
            st.session_state.current_index = 0
            st.session_state.interview_data = []
            st.success("Shortlisting complete! You can now proceed to the voice interview.")

    # 2️⃣ Voice Interview
    if st.session_state.get("candidates"):
        interview_app()

def interview_app():
    questions = [
        "What position are you applying for?",
        "Can you briefly describe your previous work experience?",
        "What is your highest qualification?",
        "Are you available for remote work?",
        "What are your salary expectations?"
    ]

    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "step" not in st.session_state:
        st.session_state.step = 0
        st.session_state.answers = {}
        st.session_state.greeted = False
    if "interview_data" not in st.session_state:
        st.session_state.interview_data = []

    candidates = st.session_state.candidates
    idx = st.session_state.current_index

    if idx >= len(candidates):
        st.success("✅ All interviews complete!")
        df = pd.DataFrame(st.session_state.interview_data)
        st.download_button("📥 Download Interview Results", df.to_csv(index=False).encode(), "interview_results.csv", mime="text/csv")
        return

    candidate = candidates[idx]
    display_name = candidate['Name'] if candidate['Name'].strip() else "candidate"
    st.subheader(f"Interviewing: {display_name}")

    if st.session_state.step == 0 and not st.session_state.greeted:
        greeting = f"Hello {display_name}, welcome to your interview."
        play_audio(greeting)
        st.session_state.greeted = True

    if st.session_state.step < len(questions):
        q = questions[st.session_state.step]
        st.markdown(f"### ❓ {q}")

        if "last_asked" not in st.session_state or st.session_state.last_asked != q:
            play_audio(q)
            st.session_state.last_asked = q

        if st.button("🎙 Record Response"):
            audio = record_audio()
            text = transcribe_audio(audio)
            st.write(f"🗣 You said: **{text}**")
            valid = validate_answer(q, text)
            st.write(f"✅ Validity: {valid}")

            # More flexible validity check
            if "valid" in valid.lower() and "invalid" not in valid.lower():
                st.session_state.answers[q] = text
                st.session_state.step += 1
                st.session_state.last_asked = None
                st.rerun()
            else:
                st.warning("Answer is not Valid. Please provide a valid answer.")
                play_audio("Answer is not valid. Please provide a valid answer. " + q)
        
    else:
        score, details = score_interview(st.session_state.answers)
        sentiment, sent_details = analyze_sentiment(list(st.session_state.answers.values()))
        st.markdown(f"### 🎯 Score: {score}\n\n**Details:** {details}")
        st.markdown(f"### 😊 Sentiment: {sentiment}")
        st.markdown(f"**Sentiment Details:** {sent_details}")

        st.session_state.interview_data.append({
            "Name": display_name,
            "Score": score,
            "Sentiment": sentiment,
            "Details": details
        })

        if st.button("➡️ Next Candidate"):
            st.session_state.current_index += 1
            st.session_state.step = 0
            st.session_state.answers = {}
            st.session_state.greeted = False
            st.session_state.last_asked = None
            st.rerun()


if __name__ == "__main__":
    main()
