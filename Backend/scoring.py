import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def evaluate_response(transcript):
    """Evaluates IELTS speaking response using GPT-4."""
    prompt = f"""
    Evaluate the following IELTS Speaking response based on:
    1️⃣ Fluency & Coherence
    2️⃣ Lexical Resource
    3️⃣ Grammatical Range & Accuracy
    4️⃣ Pronunciation (if possible from text)
    
    Provide a score (0-9) and specific feedback for each.

    Response: {transcript}
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"]
