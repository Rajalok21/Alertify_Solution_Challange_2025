import os
from pydub import AudioSegment

def convert_to_wav(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    audio = AudioSegment.from_file(file_path)

    output_file_path = os.path.splitext(file_path)[0] + ".wav"

    audio.export(output_file_path, format="wav")
    return output_file_path