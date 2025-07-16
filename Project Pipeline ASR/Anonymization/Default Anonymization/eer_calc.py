import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import random
from tqdm import tqdm # For progress bar

# Set environment variable to suppress symlink warning from Hugging Face Hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
print("Environment variable HF_HUB_DISABLE_SYMLINKS_WARNING set to 1 for this process.")

# --- Configuration ---
# IMPORTANT: Adjust these paths to your actual setup
PROJECT_ROOT_PATH = "C:/Users/Hans Roozen/Documents/Programming/ASR_Project"

# Path to your master CSV for LibriSpeech test-clean
METADATA_CSV_PATH = os.path.join(PROJECT_ROOT_PATH, "librispeech_test_metadata.csv")

# Base directories for different datasets
# ORIGINAL_DATA_DIR is now PROJECT_ROOT_PATH because the CSV's relative_path
# likely includes 'test-clean/' (e.g., 'test-clean/speaker_id/...')
ORIGINAL_DATA_DIR = PROJECT_ROOT_PATH 
ANONYMIZED_DATA_ROOT = os.path.join(PROJECT_ROOT_PATH, "data") # Contains Anon_data_McAdams_Fixed etc.

# Anonymization folder names and suffixes
FIXED_ANON_FOLDER = 'Anon_data_McAdams_Fixed'
FIXED_ANON_SUFFIX = '_fixed_0.80'
DYNAMIC_ANON_FOLDER = 'Anon_data_McAdams_Dynamic'
DYNAMIC_ANON_SUFFIX = '_dynamic_0.70_0.90'
RANDOM_ANON_FOLDER = 'Anon_data_McAdams_Random'
RANDOM_ANON_SUFFIX = '_random_uniform'

# Parameters for EER calculation
M_UTTERANCES_PER_SPEAKER = 8  # Number of utterances per speaker (M/2 for enrollment, M/2 for verification)
NUM_EER_EPOCHS = 100          # Number of repetitions for EER calculation (to average results)

# --- Initialize Speaker Embedding Model (for EER) ---
# Load a pre-trained x-vector model from SpeechBrain
# This model is trained on VoxCeleb and is suitable for speaker verification.
try:
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb"
    )
    print("SpeechBrain x-vector model loaded successfully.")
except Exception as e:
    print(f"Error loading SpeechBrain model: {e}")
    print("Please ensure you have an active internet connection or the model is cached locally.")
    print("You might need to install speechbrain: pip install speechbrain torchaudio")
    exit()

# --- Helper Functions ---

def load_audio(file_path, target_sample_rate=16000):
    """
    Loads an audio file and resamples it to the target_sample_rate.
    Returns a torch tensor.
    """
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
        # SpeechBrain models typically expect mono audio
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0) # Remove channel dimension if mono
    except Exception as e:
        # print(f"Error loading or processing audio file {file_path}: {e}") # Suppress frequent warnings
        return None

def get_embedding(audio_tensor):
    """
    Extracts a speaker embedding from an audio tensor using the pre-trained model.
    """
    # Ensure audio_tensor is 2D: (batch, samples) for encode_batch
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0) # (1, samples)
    elif audio_tensor.dim() == 2 and audio_tensor.shape[0] > 1:
        # If it's (channels, samples) and has multiple channels, convert to mono
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True) # (1, samples)
    elif audio_tensor.dim() == 2 and audio_tensor.shape[0] == 1:
        # If it's already (1, samples), keep as is
        pass
    else:
        raise ValueError(f"Unsupported audio tensor dimension: {audio_tensor.dim()}. Expected 1D or 2D.")

    # Ensure the tensor is on the correct device (CPU/GPU)
    audio_tensor = audio_tensor.to(classifier.device)

    # Compute the embedding
    with torch.no_grad():
        embedding = classifier.encode_batch(audio_tensor)
    return embedding.squeeze(0).squeeze(0) # Remove batch and potential extra dimensions

def cosine_similarity(emb1, emb2):
    """
    Calculates cosine similarity between two embeddings.
    """
    return torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()

def calculate_eer(genuine_scores, imposter_scores):
    """
    Calculates the Equal Error Rate (EER) given genuine and imposter scores.
    """
    scores = np.concatenate([genuine_scores, imposter_scores])
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(imposter_scores))])

    # Handle cases where there might be no genuine or imposter scores
    if len(np.unique(labels)) < 2:
        # print("Not enough unique labels to compute ROC curve (need both genuine and imposter scores).") # Suppress frequent warnings
        return 1.0 # Return a high EER if calculation is not possible

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # Find the EER point where FAR (fpr) and FRR (fnr) are closest
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except ValueError:
        # If brentq fails (e.g., due to non-monotonicity or boundary issues),
        # fall back to finding the minimum absolute difference.
        min_diff_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[min_diff_idx] + fnr[min_diff_idx]) / 2
        # print("Warning: Brentq failed for EER calculation, falling back to min absolute difference.") # Suppress frequent warnings

    return eer

def get_file_path(base_dir, relative_path_in_csv, anonymization_suffix=None, target_audio_extension=".flac"):
    """
    Constructs the full file path based on dataset type and anonymization.
    """
    if anonymization_suffix:
        parts = relative_path_in_csv.split(os.sep) 
        
        if len(parts) >= 4: # Expecting at least test-clean/speaker/chapter/file
            test_clean_dir = parts[0] 
            speaker_id_dir = parts[1] 
            chapter_id_dir = parts[2] 
            filename_with_ext = parts[3] 

            new_chapter_id_dir = f"{chapter_id_dir}{anonymization_suffix}"
            filename_base = os.path.splitext(filename_with_ext)[0]
            new_filename_with_ext = f"{filename_base}{target_audio_extension}"
            
            actual_relative_path = os.path.join(test_clean_dir, speaker_id_dir, new_chapter_id_dir, new_filename_with_ext)
        else:
            # Fallback if path format is unexpected for anonymized data
            # print(f"Warning: Unexpected relative path format for anonymized data: {relative_path_in_csv}. Using original path and only changing extension.") # Suppress frequent warnings
            actual_relative_path = os.path.splitext(relative_path_in_csv)[0] + target_audio_extension
    else:
        # For original data, just ensure correct extension if needed
        actual_relative_path = os.path.splitext(relative_path_in_csv)[0] + target_audio_extension
    
    return os.path.join(base_dir, actual_relative_path)


# --- EER Calculation Logic (Revised to match described method) ---

def run_eer_evaluation(metadata_path, data_root_dir, m_utterances_per_speaker, num_eer_epochs, anonymization_suffix=None, target_audio_extension=".flac"):
    """
    Runs the EER evaluation for a given dataset, matching the described method.
    
    Args:
        metadata_path (str): Path to the CSV metadata file.
        data_root_dir (str): Root directory where audio files are stored.
        m_utterances_per_speaker (int): Number of utterances per speaker for enrollment/verification.
        num_eer_epochs (int): Number of repetitions for EER calculation.
        anonymization_suffix (str, optional): The suffix to append to chapter IDs and filenames
                                              for anonymized data (e.g., "_fixed_0.80").
                                              Defaults to None for original data.
        target_audio_extension (str): The expected file extension for audio files in this dataset
                                      (e.g., ".flac" or ".wav").
    """
    print(f"\n--- Starting EER evaluation for data in: {data_root_dir} ---")
    df = pd.read_csv(metadata_path)

    # Filter out speakers with fewer than M_UTTERANCES_PER_SPEAKER
    speaker_counts = df['speaker_id'].value_counts()
    eligible_speakers = speaker_counts[speaker_counts >= m_utterances_per_speaker].index.tolist()

    if not eligible_speakers:
        print(f"No speakers found with at least {m_utterances_per_speaker} utterances in {data_root_dir}.")
        return None

    print(f"Found {len(eligible_speakers)} eligible speakers.")

    final_eer_results = []

    for epoch in tqdm(range(num_eer_epochs), desc="EER Epochs"):
        # Randomly select a subset of eligible speakers for this epoch
        # To ensure sufficient imposter pairs, we need at least 2 speakers.
        if len(eligible_speakers) < 2:
            # print(f"Skipping epoch {epoch + 1}: Not enough eligible speakers for imposter comparisons.") # Suppress frequent warnings
            continue

        # Select a random subset of speakers for this epoch to manage computation time
        # and ensure variability in EER calculation.
        # Limit to 20 speakers for a reasonable runtime, adjust as needed.
        selected_speakers = random.sample(eligible_speakers, min(len(eligible_speakers), 20)) 
        if len(selected_speakers) < 2:
            # print(f"Skipping epoch {epoch + 1}: Not enough speakers selected for imposter comparisons.") # Suppress frequent warnings
            continue

        # --- Load and Process Embeddings for Selected Speakers ---
        speaker_all_utterance_embeddings = {}
        for speaker_id in selected_speakers:
            current_speaker_df = df[df['speaker_id'] == speaker_id]
            
            if len(current_speaker_df) < m_utterances_per_speaker:
                # print(f"Warning: Speaker {speaker_id} has only {len(current_speaker_df)} utterances, but {m_utterances_per_speaker} are required. Skipping this speaker for epoch {epoch + 1}.") # Suppress frequent warnings
                continue

            # Sample M utterances for the current speaker
            speaker_df_sampled = current_speaker_df.sample(n=m_utterances_per_speaker, random_state=epoch) 
            
            embeddings_for_speaker = []
            for index, row in speaker_df_sampled.iterrows():
                full_path = get_file_path(data_root_dir, row['relative_path'], anonymization_suffix, target_audio_extension)
                audio = load_audio(full_path)
                if audio is not None:
                    emb = get_embedding(audio)
                    embeddings_for_speaker.append(emb)
            
            if len(embeddings_for_speaker) == m_utterances_per_speaker:
                # Store as a tensor for easier splitting and mean calculation
                speaker_all_utterance_embeddings[speaker_id] = torch.stack(embeddings_for_speaker)
            else:
                # print(f"Warning: Could not get {m_utterances_per_speaker} embeddings for speaker {speaker_id} after loading. Skipping.") # Suppress frequent warnings
                continue
        
        # Ensure we have at least two speakers with enough embeddings for comparisons
        if len(speaker_all_utterance_embeddings) < 2:
            # print(f"Skipping epoch {epoch + 1}: Not enough speakers with valid embeddings for comparisons after loading.") # Suppress frequent warnings
            continue

        # Convert dictionary of embeddings to a list of tensors for consistent indexing
        # and create a mapping from list index back to speaker_id
        ordered_speaker_ids = list(speaker_all_utterance_embeddings.keys())
        dvector_loader = torch.stack([speaker_all_utterance_embeddings[sid] for sid in ordered_speaker_ids])

        # Split into enrollment and verification embeddings
        assert M_UTTERANCES_PER_SPEAKER % 2 == 0
        enrollment_embeddings, verification_embeddings = torch.split(dvector_loader, int(M_UTTERANCES_PER_SPEAKER // 2), dim=1)

        num_speakers, num_utterances_enrollment, _ = enrollment_embeddings.shape
        _, num_utterances_verification, _ = verification_embeddings.shape

        # Calculate speaker models (mean of enrollment embeddings)
        if num_utterances_enrollment == 1:
            speaker_models = enrollment_embeddings.squeeze(1)
        else:
            speaker_models = torch.mean(enrollment_embeddings, dim=1)

        # Initialize similarity and label arrays
        # similarities will store all comparison scores (genuine and imposter)
        # labels will store 1 for genuine, 0 for imposter
        similarities = []
        labels = []

        # Calculate similarities between speaker models and verification utterances
        for i in range(num_speakers): # Iterate through enrollment speaker models (i.e., target speakers)
            for j in range(num_speakers): # Iterate through verification speakers
                for k in range(num_utterances_verification): # Iterate through verification utterances for speaker j
                    vec1 = speaker_models[i] # Target speaker model
                    vec2 = verification_embeddings[j, k] # Verification utterance from speaker j

                    similarity = torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()
                    
                    similarities.append(similarity)
                    if i == j: # If the target speaker model and verification utterance are from the same speaker
                        labels.append(1) # Genuine pair
                    else:
                        labels.append(0) # Imposter pair

        # Convert to numpy arrays for sklearn's roc_curve
        similarities_np = np.array(similarities)
        labels_np = np.array(labels)

        # Calculate EER
        current_eer = calculate_eer(similarities_np[labels_np == 1], similarities_np[labels_np == 0])
        final_eer_results.append(current_eer)

    if final_eer_results:
        mean_EER = np.mean(final_eer_results)
        std_EER = np.std(final_eer_results)
        print(f"\n--- EER Calculation Complete ---")
        print(f"Average EER over {num_eer_epochs} epochs: {mean_EER:.4f}")
        print(f"Standard Deviation of EER: {std_EER:.4f}")
        return mean_EER, std_EER
    else:
        print("\nNo EER results could be computed. Please check your data and configuration.")
        return None, None

# --- Run EER for Original Data ---
print("\nStarting EER calculation for ORIGINAL data...")
original_eer_mean, original_eer_std = run_eer_evaluation(METADATA_CSV_PATH, ORIGINAL_DATA_DIR, M_UTTERANCES_PER_SPEAKER, NUM_EER_EPOCHS, 
                                  anonymization_suffix=None, target_audio_extension=".flac")

# --- Run EER for Fixed Anonymized Data ---
print("\nStarting EER calculation for FIXED ANONYMIZED data...")
fixed_anon_data_dir = os.path.join(ANONYMIZED_DATA_ROOT, FIXED_ANON_FOLDER)
fixed_anon_eer_mean, fixed_anon_eer_std = run_eer_evaluation(METADATA_CSV_PATH, fixed_anon_data_dir, M_UTTERANCES_PER_SPEAKER, NUM_EER_EPOCHS, 
                                    anonymization_suffix=FIXED_ANON_SUFFIX, target_audio_extension=".wav")

print("\nStarting EER calculation for DYNAMIC ANONYMIZED data...")
dynamic_anon_data_dir = os.path.join(ANONYMIZED_DATA_ROOT, DYNAMIC_ANON_FOLDER)
dynamic_anon_eer_mean, dynamic_anon_eer_std = run_eer_evaluation(METADATA_CSV_PATH, dynamic_anon_data_dir, M_UTTERANCES_PER_SPEAKER, NUM_EER_EPOCHS, 
                                      anonymization_suffix=DYNAMIC_ANON_SUFFIX, target_audio_extension=".wav")

print("\nStarting EER calculation for RANDOM ANONYMIZED data...")
random_anon_data_dir = os.path.join(ANONYMIZED_DATA_ROOT, RANDOM_ANON_FOLDER)
random_anon_eer_mean, random_anon_eer_std = run_eer_evaluation(METADATA_CSV_PATH, random_anon_data_dir, M_UTTERANCES_PER_SPEAKER, NUM_EER_EPOCHS, 
                                      anonymization_suffix=RANDOM_ANON_SUFFIX, target_audio_extension=".wav")

# --- Print Summary of Results ---
print("\n--- Summary of Results ---")
if original_eer_mean is not None:
    print(f"Original Data EER: {original_eer_mean:.4f} (Std: {original_eer_std:.4f})")
if fixed_anon_eer_mean is not None:
    print(f"Fixed Anonymized Data EER: {fixed_anon_eer_mean:.4f} (Std: {fixed_anon_eer_std:.4f})")
if dynamic_anon_eer_mean is not None:
    print(f"Dynamic Anonymized Data EER: {dynamic_anon_eer_mean:.4f} (Std: {dynamic_anon_eer_std:.4f})")
if random_anon_eer_mean is not None:
    print(f"Random Anonymized Data EER: {random_anon_eer_mean:.4f} (Std: {random_anon_eer_std:.4f})")
