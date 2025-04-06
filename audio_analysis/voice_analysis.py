import extract_sound_from_videio
import mp3towav
import speech
import transcription
#extracting the sound from the video
video_path = r'C:\Users\samsung\Desktop\final\FINAL_RUN\audio_analysis\qqq.mp4'
audio_path = 'output_audio.mp3'

extract_sound_from_videio.extract_audio(video_path, audio_path)

#converting to wav
audio_path = mp3towav.convert_to_wav(audio_path)

#predicts the category of the sound and gives the related plots
category = speech.predict_and_plot(audio_path)
print(f"Predicted sound: {category}")

#transcribes the audio
language, transcript, translated_text = transcription.process_audio(audio_path)

print(language)
#print(transcript)
print(translated_text)
