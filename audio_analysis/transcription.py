import subprocess
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def process_audio(file_path, model='medium'):
    """Transcribe audio and translate the transcription to English using Gemini."""

    # Transcribe the audio using Whisper
    result = subprocess.run(
        ['whisper', file_path, '--model', model],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'
    )

    lines = result.stdout.splitlines()
    language = ""
    transcription = ""

    # Extract language and transcription
    for line in lines:
        if line.startswith("Detected language:"):
            language = line.split(":")[1].strip()
        elif "-->" in line:
            try:
                transcription += line.split("]")[1].strip() + " "
            except IndexError:
                continue

    # Translate using Gemini
    def translate_with_gemini(text):
        try:
            model = genai.GenerativeModel('gemini-pro')
            prompt = f"Translate the following text to English:\n\n{text}"
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error during Gemini translation: {str(e)}"

    translated_text = translate_with_gemini(transcription)

    return language, transcription.strip(), translated_text.strip()

