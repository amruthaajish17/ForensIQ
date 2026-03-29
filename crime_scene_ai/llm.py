# llm.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-2.5-flash")  # fast + free tier available


def generate_report(scene_text: str, retrieved_knowledge: list[str]) -> str:
    """
    Send scene description + retrieved forensic knowledge to Gemini.
    Returns a structured investigation report.
    """
    context = "\n\n".join(
        f"[Source {i+1}]: {chunk}"
        for i, chunk in enumerate(retrieved_knowledge)
    )

    prompt = f"""You are an expert AI forensic investigator assistant.

━━━ CRIME SCENE DESCRIPTION ━━━
{scene_text}

━━━ RETRIEVED FORENSIC KNOWLEDGE ━━━
{context}

Based on the scene and forensic knowledge above, generate a structured report:

## 🔍 Key Evidence Identified
List the most significant items and what they suggest.

## ⚠️ Immediate Actions (Next 30 Minutes)
Prioritized steps the first responders must take right now.

## 🧪 Recommended Forensic Tests
Specific lab tests or field tests that should be run.

## 📋 Investigation Priority Score
Score from 1–10 with brief justification.

## 🔗 Possible Scenario
One or two plausible interpretations of what may have occurred.

Be concise, professional, and actionable."""

    response = model.generate_content(prompt)
    return response.text