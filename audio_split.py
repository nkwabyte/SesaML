from pydub import AudioSegment
import math
import os
from tkinter import Tk, filedialog

def split_audio(file_path, chunk_length_sec=10, output_folder="output_chunks"):
    """
    Splits an audio file into smaller chunks.

    Args:
        file_path (str): Path to the input audio file.
        chunk_length_sec (int): Duration of each chunk in seconds.
        output_folder (str): Folder to save the chunks.
    """
    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    total_length_ms = len(audio)
    chunk_length_ms = chunk_length_sec * 1000  # convert to milliseconds

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Calculate the number of chunks
    num_chunks = math.ceil(total_length_ms / chunk_length_ms)

    print(f"Splitting into {num_chunks} chunks of {chunk_length_sec} seconds each...")

    for i in range(num_chunks):
        start_ms = i * chunk_length_ms
        end_ms = min(start_ms + chunk_length_ms, total_length_ms)

        chunk = audio[start_ms:end_ms]
        chunk_name = os.path.join(output_folder, f"chunk_{i+1}.wav")
        chunk.export(chunk_name, format="wav")
        print(f"Saved {chunk_name}")

    print("✅ Splitting completed!")


if __name__ == "__main__":
    # Hide Tkinter main window
    Tk().withdraw()

    print("📂 Please select the audio file...")
    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a")]
    )

    if file_path:
        # Ask for chunk length
        try:
            chunk_length = int(input("Enter chunk length in seconds (e.g., 10): "))
        except ValueError:
            print("Invalid input. Using default of 10 seconds.")
            chunk_length = 10

        split_audio(file_path, chunk_length_sec=chunk_length, output_folder="audio_chunks")
    else:
        print("❌ No file selected. Exiting.")
