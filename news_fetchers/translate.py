from deep_translator import GoogleTranslator

def translate_to_english(text):
    try:
        translator = GoogleTranslator(source='auto', target='en')
        return translator.translate(text)
    except Exception as e:
        return f"Error: {str(e)}"
