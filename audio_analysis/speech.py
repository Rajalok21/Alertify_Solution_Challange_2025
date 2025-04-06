import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings
import tensorflow_hub as hub
import io
import base64
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

warnings.filterwarnings("ignore")

# Load YAMNet model from TensorFlow Hub
yamnet_model = hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')

# Load class names for YAMNet
class_names_df = pd.read_csv(r'C:\Users\samsung\Desktop\final\FINAL_RUN\audio_analysis\yamnet_class_map.csv')
class_names = class_names_df.iloc[:, 2].values

def load_wav_16k_mono(filename):
    """Load an audio file and resample it to 16 kHz mono."""
    waveform, sample_rate = librosa.load(filename, sr=16000, mono=True)
    return waveform, sample_rate

def plot_audio_features(waveform, sample_rate):
    """Plot various audio features and return as base64-encoded images."""
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Audio Feature Visualization', fontsize=16)
    cmap = 'plasma'

    # Plot Sound Wave
    axs[0, 0].plot(np.arange(len(waveform)) / sample_rate, waveform, color='blue')
    axs[0, 0].set_title('Sound Wave')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].grid(True)

    # Plot Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)
    img = librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log', ax=axs[0, 1], cmap=cmap)
    axs[0, 1].set_title('Spectrogram')
    fig.colorbar(img, ax=axs[0, 1], format='%+2.0f dB')

    # Plot Mel Spectrogram
    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=waveform, sr=sample_rate), ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sample_rate, x_axis='time', y_axis='mel', ax=axs[1, 0], cmap=cmap)
    axs[1, 0].set_title('Mel Spectrogram')
    fig.colorbar(img, ax=axs[1, 0], format='%+2.0f dB')

    # Plot Chroma Features
    chroma = librosa.feature.chroma_stft(y=waveform, sr=sample_rate)
    img = librosa.display.specshow(chroma, sr=sample_rate, x_axis='time', y_axis='chroma', ax=axs[1, 1], cmap=cmap)
    axs[1, 1].set_title('Chroma Features')
    fig.colorbar(img, ax=axs[1, 1])

    # Plot Constant-Q Transform (CQT)
    cqt_db = librosa.amplitude_to_db(np.abs(librosa.cqt(waveform, sr=sample_rate)), ref=np.max)
    img = librosa.display.specshow(cqt_db, sr=sample_rate, x_axis='time', y_axis='cqt_hz', ax=axs[2, 0], cmap=cmap)
    axs[2, 0].set_title('Constant-Q Transform (CQT)')
    fig.colorbar(img, ax=axs[2, 0], format='%+2.0f dB')

    # Plot Tempogram
    tempo = librosa.feature.tempogram(y=waveform, sr=sample_rate)
    img = librosa.display.specshow(tempo, sr=sample_rate, x_axis='time', ax=axs[2, 1], cmap=cmap)
    axs[2, 1].set_title('Tempogram')
    fig.colorbar(img, ax=axs[2, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plot to a bytes buffer and encode as base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    return image_base64

def predict_and_plot(file_path):
    """Predict the category of a sound and return base64-encoded image of plots."""
    waveform, sample_rate = load_wav_16k_mono(file_path)

    # Predict category using YAMNet
    scores, embeddings, spectrogram = yamnet_model(waveform)
    top_class = np.argmax(scores, axis=1)
    top_class_name = class_names[top_class[0]]

    image_base64 = plot_audio_features(waveform, sample_rate)

    return top_class_name, image_base64
