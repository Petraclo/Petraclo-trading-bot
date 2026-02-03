import streamlit as st
from groq import Groq

api_key = st.secrets["GROQ_API_KEY"]  # safely loaded from secrets.toml
client = Groq(api_key=api_key)

SYSTEM_PROMPT = """You are a trading strategy analysis assistant.
You answer based only on provided backtest results.
Suggest parameter adjustments, highlight risks, and give neutral recommendations based on user needs.
Never predict the market. Do not give financial advice; provide insights only.
"""

def ask_groq(user_prompt, context_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt + "\n\nCONTEXT:\n" + context_text}
    ]
    resp = client.chat.completions.create(
        model="llama3-70b-8192",  # Groq-supported model
        messages=messages,
        temperature=0.2,
        max_tokens=800,
    )
    return resp.choices[0].message.content
