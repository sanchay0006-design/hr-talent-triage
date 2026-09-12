import json
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# Page Configuration
st.set_page_config(
    page_title="Agentic Talent Triage | Horizon Group",
    page_icon="💼",
    layout="wide",
)

st.title("💼 Horizon Talent Triage & Candidate Engagement Engine")
st.caption(
    "Explainable AI screening and automated empathetic candidate communication."
)

# Sidebar: Configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Enter Google Gemini API Key",
        type="password",
        help="Get your free key from aistudio.google.com",
    )
    st.markdown("---")
    st.markdown(
        "**Context:** Solves the 240,000 applicant bottleneck by providing 100% coverage, "
        "explainable scoring, and zero candidate ghosting."
    )

# Main Screen Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Job Description")
    jd_input_method = st.radio(
        "Choose JD input method:",
        ["Upload JD (PDF)", "Paste JD Text"],
        horizontal=True,
        key="jd_radio"
    )

    jd_file = None
    jd_text_input = ""

    if jd_input_method == "Upload JD (PDF)":
        jd_file = st.file_uploader(
            "Upload Job Description (PDF only)", type=["pdf"], key="jd_upload"
        )
    else:
        jd_text_input = st.text_area(
            "Paste Job Description text directly:",
            height=180,
            placeholder="Paste role requirements here...",
        )

with col2:
    st.subheader("2. Candidate Resumes")
    resume_input_method = st.radio(
        "Choose Resume input method:",
        ["Upload Resumes (Batch PDF)", "Paste Resume Text"],
        horizontal=True,
        key="resume_radio"
    )

    uploaded_files = []
    pasted_resume = ""

    if resume_input_method == "Upload Resumes (Batch PDF)":
        uploaded_files = st.file_uploader(
            "Upload candidate resumes (PDFs)", type=["pdf"], accept_multiple_files=True
        )
    else:
        pasted_resume = st.text_area(
            "Paste resume text directly:",
            height=180,
            placeholder="Paste candidate work experience, skills, and education here...",
        )


def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    extracted = ""
    for page in reader.pages:
        extracted += page.extract_text() or ""
    return extracted


# Evaluation Trigger
if st.button("Evaluate Batch & Generate Responses", type="primary"):
    
    # 1. Resolve JD Text
    jd_text = ""
    if jd_input_method == "Upload JD (PDF)" and jd_file:
        try:
            jd_text = extract_text_from_pdf(jd_file)
        except Exception as e:
            st.error(f"Failed to read JD PDF: {e}")
    elif jd_input_method == "Paste JD Text" and jd_text_input.strip():
        jd_text = jd_text_input.strip()

    # 2. Validations
    if not api_key:
        st.error("Please provide a Gemini API Key in the left sidebar.")
    elif not jd_text:
        st.warning("Please upload or paste a Job Description to proceed.")
    elif resume_input_method == "Upload Resumes (Batch PDF)" and not uploaded_files:
        st.warning("Please upload at least one resume PDF to proceed.")
    elif resume_input_method == "Paste Resume Text" and not pasted_resume.strip():
        st.warning("Please paste resume text to proceed.")
    else:
        # 3. Prepare the list of candidates to process
        candidates_to_process = []
        if resume_input_method == "Upload Resumes (Batch PDF)":
            for file in uploaded_files:
                try:
                    text = extract_text_from_pdf(file)
                    candidates_to_process.append({"name": file.name, "text": text})
                except Exception as e:
                    st.error(f"Failed to read {file.name}: {e}")
        else:
            candidates_to_process.append({"name": "Pasted Candidate", "text": pasted_resume.strip()})

        # 4. Initialize AI Client & Process
        try:
            client = genai.Client(api_key=api_key)
            system_instruction = (
                "You are an expert Talent Acquisition Assessor for Horizon Group. "
                "Your objective is to conduct an objective, unbiased evaluation of the candidate "
                "against the job requirements. Evaluate transferable capabilities and semantic context, "
                "not just exact keywords. You must output strictly valid JSON matching the requested schema."
            )

            st.success(f"Processing batch of {len(candidates_to_process)} candidate(s)...")
            st.markdown("---")

            for idx, candidate in enumerate(candidates_to_process):
                st.subheader(f"📄 Evaluating: {candidate['name']}")
                
                with st.spinner(f"Analyzing {candidate['name']} and drafting communication..."):
                    prompt = f"""
Job Description:
\"\"\"
{jd_text}
\"\"\"

Candidate Resume Text:
\"\"\"
{candidate['text']}
\"\"\"

Analyze the candidate against the role requirements and return a JSON object with EXACTLY these keys:
- "candidate_name": The extracted name of the candidate.
- "match_score": An integer from 0 to 100 indicating fit against the role competencies.
- "fit_rationales": A list of 3 specific bullet points explaining why the candidate received this score.
- "skill_gaps": A list of 1 to 2 identified competency gaps relative to the job requirements.
- "candidate_email": A professionally written, highly empathetic email to the candidate. 
    If match_score >= 70: Draft an interview invitation mentioning specific strengths from their resume.
    If match_score < 70: Draft a respectful, personalized rejection that explains the gap constructive to their growth, ensuring they feel evaluated rather than ignored.
"""
                    try:
                        response = client.models.generate_content(
                            model="gemini-3.6-flash",
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction,
                                response_mime_type="application/json",
                                temperature=0.2,
                            ),
                        )

                        result = json.loads(response.text)

                        res_col1, res_col2 = st.columns([1, 2])

                        with res_col1:
                            st.metric(
                                label="Match Score",
                                value=f"{result.get('match_score', 0)}/100",
                            )
                            st.write(
                                f"**Name:** {result.get('candidate_name', 'N/A')}"
                            )

                            if result.get("match_score", 0) >= 70:
                                st.success("Recommendation: Shortlist")
                            else:
                                st.info("Recommendation: Archive / Nurture")

                        with res_col2:
                            st.write("**Explainable AI Fit Assessment:**")
                            for point in result.get("fit_rationales", []):
                                st.write(f"- {point}")

                            if result.get("skill_gaps"):
                                st.write("**Identified Capability Gaps:**")
                                for gap in result.get("skill_gaps", []):
                                    st.write(f"- ⚠️ {gap}")

                        st.markdown("**Automated Empathetic Communication:**")
                        st.text_area(
                            f"Draft Email for {result.get('candidate_name', 'Candidate')}:",
                            value=result.get("candidate_email", ""),
                            height=200,
                            key=f"email_{idx}" 
                        )

                    except Exception as e:
                        st.error(f"AI evaluation failed for {candidate['name']}: {e}")
                
                st.markdown("---")

        except Exception as e:
            st.error(f"Failed to initialize Gemini Client: {e}")