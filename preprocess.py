import os
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from tqdm import tqdm

INPUT_PATH = "dataset"
OUTPUT_PATH = "processed"

SAMPLE_RATE = 16000


# Convert any file to wav
def convert_to_wav(input_path):
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)  # mono
        return audio
    except:
        return None


# Normalize audio
def normalize_audio(audio):
    if np.max(np.abs(audio)) != 0:
        audio = audio / np.max(np.abs(audio))
    return audio


# Process single file
def process_file(input_file, output_file):
    try:
        # Convert to wav using pydub
        audio = convert_to_wav(input_file)
        if audio is None:
            return

        temp_wav = "temp.wav"
        audio.export(temp_wav, format="wav")

        # Load with librosa
        signal, sr = librosa.load(temp_wav, sr=SAMPLE_RATE)

        # Normalize
        signal = normalize_audio(signal)

        # Noise reduction
        signal = nr.reduce_noise(y=signal, sr=sr)

        # Trim silence
        signal, _ = librosa.effects.trim(signal)

        # Save
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        sf.write(output_file, signal, sr)

    except Exception as e:
        print("Error:", input_file)


# Process dataset
def process_dataset():
    for split in ["train", "test"]:
        for label in ["real", "fake"]:

            input_folder = os.path.join(INPUT_PATH, split, label)
            output_folder = os.path.join(OUTPUT_PATH, split, label)

            print(f"\nProcessing {split}/{label}...")

            if not os.path.exists(input_folder):
                print("Folder not found:", input_folder)
                continue

            for file in tqdm(os.listdir(input_folder)):

                input_file = os.path.join(input_folder, file)

                # Only valid audio files
                if not file.lower().endswith((".wav", ".mp3", ".flac")):
                    continue

                filename = os.path.splitext(file)[0] + ".wav"
                output_file = os.path.join(output_folder, filename)

                process_file(input_file, output_file)


# Run
if __name__ == "__main__":
    process_dataset()
    print("\n✅ Preprocessing completed successfully!")