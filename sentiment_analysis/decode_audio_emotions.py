import os
import pandas as pd
from pathlib import Path
import json

def decode_ravdess_filename(filename):
    """
    Decode a RAVDESS filename to extract metadata.
    Format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav

    Args:
        filename (str): File name (e.g., "03-01-05-02-01-01-12.wav")

    Returns:
        dict: Dictionary with the decoded metadata
    """

    # RAVDESS mappings
    emotion_mapping = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }

    intensity_mapping = {
        '01': 'normal',
        '02': 'strong'
    }

    statement_mapping = {
        '01': 'Kids are talking by the door',
        '02': 'Dogs are sitting by the door'
    }

    # Remove the extension and split by dashes
    name_parts = filename.replace('.wav', '').split('-')

    if len(name_parts) != 7:
        return {
            'filename': filename,
            'emotion': 'unknown',
            'actor': 'unknown',
            'intensity': 'unknown',
            'valid': False
        }

    modality, vocal_channel, emotion, intensity, statement, repetition, actor = name_parts

    return {
        'filename': filename,
        'modality': modality,
        'vocal_channel': vocal_channel,
        'emotion': emotion_mapping.get(emotion, f'unknown_{emotion}'),
        'emotion_code': emotion,
        'intensity': intensity_mapping.get(intensity, f'unknown_{intensity}'),
        'intensity_code': intensity,
        'statement': statement_mapping.get(statement, f'statement_{statement}'),
        'statement_code': statement,
        'repetition': repetition,
        'actor': f"Actor_{int(actor):02d}",
        'actor_code': actor,
        'valid': True
    }

def create_audio_emotion_mapping(audio_base_dir):
    """
    Create a full mapping of all audio files with their metadata.

    Args:
        audio_base_dir (str): Path to the Audio_Speech_Actors directory

    Returns:
        dict: Dictionary {audio_path: metadata}
    """
    audio_mapping = {}
    audio_base_path = Path(audio_base_dir)

    # Loop through all Actor_XX folders
    for actor_dir in audio_base_path.glob("Actor_*"):
        if actor_dir.is_dir():
            print(f"Processing folder: {actor_dir.name}")

            # Loop through all .wav files inside the folder
            for audio_file in actor_dir.glob("*.wav"):
                relative_path = str(audio_file.relative_to(audio_base_path))
                metadata = decode_ravdess_filename(audio_file.name)
                metadata['relative_path'] = relative_path
                metadata['actor_folder'] = actor_dir.name

                audio_mapping[relative_path] = metadata

    return audio_mapping

def create_simple_emotion_dict(audio_mapping):
    """
    Create a simplified dictionary {audio_file: emotion}.

    Args:
        audio_mapping (dict): Full metadata mapping

    Returns:
        dict: Simple dictionary {filename: emotion}
    """
    return {path: data['emotion'] for path, data in audio_mapping.items()}

def save_mappings(audio_mapping, output_dir='data/processed'):
    """
    Save the mappings to different file formats.

    Args:
        audio_mapping (dict): Full metadata mapping
        output_dir (str): Directory where the files will be saved

    Returns:
        pd.DataFrame: DataFrame containing the metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Full mapping as JSON
    with open(output_path / 'audio_metadata_complete.json', 'w') as f:
        json.dump(audio_mapping, f, indent=2, ensure_ascii=False)

    # 2. Simple emotion mapping as JSON
    simple_mapping = create_simple_emotion_dict(audio_mapping)
    with open(output_path / 'audio_emotion_simple.json', 'w') as f:
        json.dump(simple_mapping, f, indent=2, ensure_ascii=False)

    # 3. CSV DataFrame for analysis
    df = pd.DataFrame.from_dict(audio_mapping, orient='index')
    df.to_csv(output_path / 'audio_metadata.csv', index=False)

    print(f"Files saved to {output_path}/")
    return df

def analyze_dataset(df):
    """
    Perform a statistical analysis of the dataset.

    Args:
        df (pd.DataFrame): DataFrame containing the metadata

    Returns:
        tuple: (emotion_counts, intensity_counts)
    """
    print("\n=== DATASET ANALYSIS ===")
    print(f"Total number of files: {len(df)}")
    print(f"Number of actors: {df['actor'].nunique()}")
    print(f"Valid files: {df['valid'].sum()}")

    print("\n--- Emotion distribution ---")
    emotion_counts = df['emotion'].value_counts()
    print(emotion_counts)

    print("\n--- Intensity distribution ---")
    intensity_counts = df['intensity'].value_counts()
    print(intensity_counts)

    print("\n--- Emotions per actor ---")
    print(f"Each actor has on average {len(df) / df['actor'].nunique():.1f} recordings")

    return emotion_counts, intensity_counts

# Main function
def main():
    # Path to your raw data
    audio_dir = "data/raw/Audio_Speech_Actors"

    print("Creating audio -> emotion mapping...")

    # Create full mapping
    audio_mapping = create_audio_emotion_mapping(audio_dir)

    # Save the results
    df = save_mappings(audio_mapping)

    # Analyze the dataset
    analyze_dataset(df)

    # Example usage
    print("\n=== EXAMPLES ===")
    simple_dict = create_simple_emotion_dict(audio_mapping)

    # Show a few examples
    for i, (audio_path, emotion) in enumerate(list(simple_dict.items())[:5]):
        print(f"{audio_path} -> {emotion}")

    print(f"\nDictionary created with {len(simple_dict)} audio files!")

    return audio_mapping, simple_dict, df

if __name__ == "__main__":
    mapping, simple_dict, dataframe = main()
