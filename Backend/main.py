from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import openai
from google.cloud import speech
import io
import base64
from config import OPENAI_API_KEY  # Load API key from config.py

app = FastAPI()

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Initialize Google Speech-to-Text Client
client = speech.SpeechClient()


class SpeechRequest(BaseModel):
    audio_base64: str  # Expecting base64-encoded audio


@app.post("/transcribe/")
async def transcribe_audio(request: SpeechRequest):
    """ Converts speech to text using Google Speech-to-Text """
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_base64)
        audio = speech.RecognitionAudio(content=audio_bytes)

        # Configure Speech Recognition
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US"
        )

        # Perform Speech Recognition
        response = client.recognize(config=config, audio=audio)

        if response.results:
            transcript = response.results[0].alternatives[0].transcript
        else:
            transcript = "Could not transcribe audio."

        return {"transcription": transcript}

    except Exception as e:
        return {"error": f"Error processing audio: {str(e)}"}


@app.websocket("/ws")
async def conversation(websocket: WebSocket):
    """ Real-time WebSocket for IELTS Speaking Simulation """
    await websocket.accept()
    await websocket.send_text("Welcome to the IELTS Speaking Test. Let's begin.")

    try:
        while True:
            data = await websocket.receive_text()
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an IELTS examiner."},
                    {"role": "user", "content": data}
                ]
            )
            reply = response["choices"][0]["message"]["content"]
            await websocket.send_text(reply)
    except WebSocketDisconnect:
        print("WebSocket connection closed.")
