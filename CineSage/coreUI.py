import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

# Page Config
st.set_page_config(page_title="Movie Information Extractor")

st.title("🎬 Movie Information Extraction")

# Model
model = ChatMistralAI(
    model="mistral-small-2603"
)

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an intelligent AI Information Extraction Assistant.

Your task is to carefully analyze the provided movie-related paragraph and extract all useful information in a clean, structured, and easy-to-read format.

Instructions:
- Be accurate and concise.
- Do not hallucinate missing facts.
- If information is unavailable, write "Not Mentioned".
- Organize the response using proper headings and bullet points.
- Extract both explicitly mentioned and strongly implied information.
- Generate a short quick summary in 2-4 lines.
- Keep the formatting neat and professional.

Extract the following information:

1. Movie Name
2. Director
3. Genre
4. Main Characters / Cast
5. Supporting Characters
6. Plot Overview
7. Main Conflict
8. Themes
9. Emotional Tone
10. Important Locations
11. Scientific / Technical Concepts
12. Important Objects or Technologies
13. Mission or Goal
14. Family Relationships
15. Survival Elements
16. Villain / Antagonist
17. Ending Type
18. Timeline or Time-related Elements
19. Language
20. Movie Type
21. Keywords / Tags
22. Quick Summary
"""
        ),
        (
            "human",
            """
Analyze the following text and extract all useful information.

Extract the information in a clean, structured, and easy-to-read format. Be accurate and concise. Do not hallucinate missing facts.

{paragraph}
"""
        )
    ]
)

# Input Box
para = st.text_area("Enter Movie Paragraph", height=250)

# Button
if st.button("Extract Information"):

    if para.strip() == "":
        st.warning("Please enter a paragraph.")
    else:
        final_prompt = prompt.invoke(
            {"paragraph": para}
        )

        response = model.invoke(final_prompt)

        st.subheader("Extracted Information")
        st.markdown(response.content)