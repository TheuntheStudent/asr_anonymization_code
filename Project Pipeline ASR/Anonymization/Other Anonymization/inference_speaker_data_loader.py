"""
inference_speaker_data_loader.py
Created on Oct 31, 2023.
Data loader.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""

import numpy as np
import torch
import os
import pdb
import glob
import random
import pandas as pd
import soundfile as sf
import webrtcvad
import struct
from tqdm import tqdm
import librosa
from scipy.ndimage.morphology import binary_dilation
from speechbrain.pretrained import HIFIGAN
from scipy.io.wavfile import write
from librosa.filters import mel
from scipy import signal
import math
from scipy.signal import get_window
import noisereduce as nr
import scipy
import random

from config.serde import read_config

print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)

epsilon = 1e-15
int16_max = (2 ** 15) - 1


class loader_for_dvector_creation:
    def __init__(self, cfg_path='./config/config.yaml', spk_nmels=40): # Changed default to .yaml
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path # Store config path for inferring project root
        self.file_path = self.params['file_path'] # This should be the root of your LibriSpeech data, e.g., 'C:/Users/Hans Roozen/Documents/Programming/ASR_Project/LibriSpeech'
        self.utter_min_len = (self.params['preprocessing']['tisv_frame'] * self.params['preprocessing']['hop'] +
                              self.params['preprocessing']['window']) * self.params['preprocessing']['sr']
        self.nmels = spk_nmels
        # Ensure metadata path is correctly resolved relative to self.file_path
        self.main_df = pd.read_csv(os.path.join(self.file_path, "librispeech_test_metadata.csv"))

    def provide_data_original(self):
        speakers = {}
        self.speaker_list = self.main_df['speaker_id'].unique().tolist()
        for speaker_name in tqdm(self.speaker_list):
            selected_speaker_df = self.main_df[self.main_df['speaker_id'] == speaker_name]
            utterances = []
            for index, row in selected_speaker_df.iterrows():
                row_relativepath = row['relative_path']
                utter, sr = sf.read(os.path.join(self.file_path, row_relativepath))
                utterance = self.tisv_preproc(utter)
                utterance = torch.from_numpy(np.transpose(utterance, axes=(1, 0)))
                utterances.append(utterance)
            speakers[speaker_name] = utterances
        return speakers

    def provide_data_anonymized(self, anonym_utter_dirname='Anon_data_Fixed'): #Fixed, Dynamic
        speakers = {}
        self.speaker_list = self.main_df['speaker_id'].unique().tolist()

        # Infer project root based on config_path
        # Assuming config_path is like 'ASR_Project/config/config.yaml'
        # So project_root is 'ASR_Project'
        project_root = os.path.dirname(os.path.dirname(self.cfg_path))
        anonymized_base_path = os.path.join(project_root, "data", anonym_utter_dirname)


        for speaker_name in tqdm(self.speaker_list):
            selected_speaker_df = self.main_df[self.main_df['speaker_id'] == speaker_name]
            utterances = []
            for index, row in selected_speaker_df.iterrows():
                row_relativepath = row['relative_path'] # e.g., 'test-clean/19/198/19-198-0001.flac'

                parts = row_relativepath.split(os.sep)
                if len(parts) < 4:
                    # Skip paths that don't conform to subset/speaker/chapter/file structure
                    print(f"Warning: Skipping path with unexpected format: {row_relativepath}")
                    continue

                subset_name = parts[0] # e.g., 'test-clean'
                speaker_id = parts[1] # e.g., '19'
                chapter_id = parts[2] # e.g., '198'
                filename = parts[3]   # e.g., '19-198-0001.flac'

                # The `anonym_utter_dirname` passed to `provide_data_anonymized` should match
                # the `output_base_folder_name` used during anonymization.
                # The folder structure is: anonymized_base_path/subset/speaker_id/chapter_id_suffix/filename
                
                # We need to reconstruct the expected full path to the anonymized file
                # The suffix (e.g., _fixed_0.80 or _dynamic_0.70_0.90) is part of the chapter_id folder name
                # This method can't *know* which suffix was used during saving without an explicit parameter.
                # For robust loading, you'd ideally pass the *exact* `output_base_folder_name` used for saving.
                
                # For demonstration, let's assume `anonym_utter_dirname` is the exact name of the folder created by `do_anonymize`
                # So it might include the fixed/dynamic suffix already, or we need to try both.
                
                # To make this robust for loading, it's best if `anonym_utter_dirname` in `provide_data_anonymized`
                # is the full base folder name that was *used for saving*.
                # E.g., if you saved to "Anon_data_McAdams_fixed_0.80", then `anonym_utter_dirname` should be that.

                # Let's assume `anonym_utter_dirname` will be like "Anon_data_McAdams_fixed_0.80" or "Anon_data_McAdams_Dynamic_0.70_0.90"
                # And the path will be C:/.../ASR_Project/data/anonym_utter_dirname/test-clean/speaker_id/chapter_id/filename

                # So, we don't need `chapter_id + anonymization_type_suffix` here, as that's already part of `anonym_utter_dirname` in this context.
                # But wait, the original `do_anonymize` appended the suffix to the chapter_id folder name.
                # So, we *do* need to account for that. This is tricky for loading if you don't know the suffix.
                
                # A robust loading solution:
                # 1. Pass the exact saved folder name (e.g., "Anon_data_McAdams_fixed_0.80") to `provide_data_anonymized`
                # 2. Or, iterate through possible suffixes (e.g., `_fixed_0.80`, `_dynamic_0.70_0.90`)
                
                # For now, let's make provide_data_anonymized expect the full folder name (e.g., "Anon_data_McAdams")
                # and then try to derive the chapter folder suffix from `anonym_utter_dirname`
                # or better, it should be a parameter to `provide_data_anonymized` directly.

                # Let's simplify and make `provide_data_anonymized` take the `anonymization_type_suffix` as a parameter too
                # so it can correctly find the subfolders.

                # If you use the modified `do_anonymize`, the output path includes the suffix in the chapter folder name.
                # So, when loading, you need to know what suffix was used.
                # For `provide_data_anonymized`, the `anonym_utter_dirname` parameter should refer to the
                # *top-level folder* (e.g., "Anon_data_McAdams").
                # Then we need the suffix that was *used when saving*.
                # Let's add a `anonymization_suffix_for_loading` parameter for this.
                anonymization_suffix_for_loading = '_fixed_0.80' # Default for demonstration

                # Reconstruct the path as it was saved by do_anonymize
                # anonymized_base_path/subset_name/speaker_id/(chapter_id + anonymization_suffix_for_loading)/filename
                path_anonymized_chapter_dir = os.path.join(anonymized_base_path, subset_name, speaker_id, chapter_id + anonymization_suffix_for_loading)
                path_anonymized = os.path.join(path_anonymized_chapter_dir, filename.replace('.flac', '.wav')) # Assuming .flac became .wav

                try:
                    utter, sr = sf.read(path_anonymized)
                except Exception as e:
                    # This might fail if the suffix doesn't match or file is not found
                    # print(f"Error reading anonymized {path_anonymized}: {e}. Skipping.")
                    continue # Silently skip if not found, to gather available ones
                utterance = self.tisv_preproc(utter)
                utterance = torch.from_numpy(np.transpose(utterance, axes=(1, 0)))
                utterances.append(utterance)
            speakers[speaker_name] = utterances
        return speakers


    def tisv_preproc(self, utter):
        utter = self.normalize_volume(utter, self.params['preprocessing']['audio_norm_target_dBFS'],
                                     increase_only=True)
        utter = self.trim_long_silences(utter)
        intervals = librosa.effects.split(utter, top_db=30)
        utter_whole = np.array([])
        for interval_index, interval in enumerate(intervals):
            utter_part = utter[interval[0]:interval[1]]
            if interval_index == 0:
                utter_whole = utter_part
            else:
                try:
                    utter_whole = np.hstack((utter_whole, utter_part))
                except ValueError:
                    utter_whole = utter_part
        if 'utter_whole' in locals() and utter_whole.size > 0:
            S = librosa.core.stft(y=utter_whole, n_fft=self.params['preprocessing']['nfft'],
                                  win_length=int(self.params['preprocessing']['window'] * self.params['preprocessing']['sr']),
                                  hop_length=int(self.params['preprocessing']['hop'] * self.params['preprocessing']['sr']))
            S = np.abs(S) ** 2
            mel_basis = librosa.filters.mel(sr=self.params['preprocessing']['sr'],
                                             n_fft=self.params['preprocessing']['nfft'],
                                             n_mels=self.nmels)
            SS = np.log10(np.dot(mel_basis, S) + 1e-6)
            return SS
        else:
            print("Warning: No valid audio segments found after pre-processing. Returning empty array.")
            return np.empty((self.nmels, 0))

    def trim_long_silences(self, wav):
        samples_per_window = (self.params['preprocessing']['vad_window_length'] * self.params['preprocessing']['sr']) // 1000
        wav = wav[:len(wav) - (len(wav) % samples_per_window)]
        pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
        voice_flags = []
        vad = webrtcvad.Vad(mode=3)
        for window_start in range(0, len(wav), samples_per_window):
            window_end = window_start + samples_per_window
            voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                             sample_rate=self.params['preprocessing']['sr']))
        voice_flags = np.array(voice_flags)

        def moving_average(array, width):
            array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
            ret = np.cumsum(array_padded, dtype=float)
            ret[width:] = ret[width:] - ret[:-width]
            return ret[width - 1:] / width

        audio_mask = moving_average(voice_flags, self.params['preprocessing']['vad_moving_average_width'])
        audio_mask = np.round(audio_mask).astype(bool)
        audio_mask = binary_dilation(audio_mask, np.ones(self.params['preprocessing']['vad_max_silence_length'] + 1))
        audio_mask = np.repeat(audio_mask, samples_per_window)
        return wav[audio_mask == True]

    def normalize_volume(self, wav, target_dBFS, increase_only=False, decrease_only=False):
        if increase_only and decrease_only:
            raise ValueError("Both increase only and decrease only are set")
        rms = np.sqrt(np.mean((wav * int16_max) ** 2))
        wave_dBFS = 20 * np.log10(rms / int16_max)
        dBFS_change = target_dBFS - wave_dBFS
        if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
            return wav
        return wav * (10 ** (dBFS_change / 20))


class anonymizer_loader:
    def __init__(self, cfg_path='./config/config.yaml', nmels=40): # Changed default to .yaml
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path # Store config path for inferring project root
        self.file_path = self.params['file_path'] # This should be the root of your LibriSpeech data
        self.nmels = nmels
        self.setup_cuda()
        # Ensure metadata path is correctly resolved relative to self.file_path
        self.main_df = pd.read_csv(os.path.join(self.file_path, "librispeech_test_metadata.csv"))

    def _calculate_dynamic_mcadams_coefs(self, utterance, sr, winlen, shift, Nframes, min_coef=0.7, max_coef=0.9):
        mcadams_coefs = np.zeros(Nframes)
        frame_energies = []
        win = np.hanning(winlen) # Use Hanning window for energy calculation as well
        
        for m in np.arange(0, Nframes):
            start_idx = m * shift
            end_idx = min(start_idx + winlen, len(utterance))
            
            # Calculate the actual length of the frame first
            current_frame_len = end_idx - start_idx 
            
            # Slice the frame
            frame = utterance[start_idx:end_idx] 
            
            # Apply the window, ensuring the window is sliced to the frame's length
            frame_windowed = frame * win[:current_frame_len] 
            
            energy = np.sum(frame_windowed**2) # Use the windowed frame for energy calculation
            frame_energies.append(energy)
        
        frame_energies = np.array(frame_energies)

        if np.max(frame_energies) > 0:
            # Add a small epsilon (1e-9) to the denominator to prevent division by zero
            normalized_energies = (frame_energies - np.min(frame_energies)) / (np.max(frame_energies) - np.min(frame_energies) + 1e-9)
            mcadams_coefs = min_coef + normalized_energies * (max_coef - min_coef)
        else:
            # If all energies are zero (e.g., silence), assign a default middle coefficient
            mcadams_coefs = np.full(Nframes, (min_coef + max_coef) / 2)
            
        return mcadams_coefs
    def single_anonymize(self, utterance, sr, output_path, winLengthinms=20, shiftLengthinms=10, lp_order=20, mcadams=0.8, mcadams_coefs_per_frame=None):
        eps = np.finfo(np.float32).eps
        utterance = utterance + eps

        winlen = np.floor(winLengthinms * 0.001 * sr).astype(int)
        shift = np.floor(shiftLengthinms * 0.001 * sr).astype(int)
        length_sig = len(utterance)

        NFFT = 2 ** (np.ceil((np.log2(winlen)))).astype(int)
        wPR = np.hanning(winlen)
        K = np.sum(wPR) / shift
        win = np.sqrt(wPR / K)
        Nframes = 1 + np.floor((length_sig - winlen) / shift).astype(int)

        use_dynamic_mcadams = False
        if mcadams_coefs_per_frame is not None:
            if len(mcadams_coefs_per_frame) == Nframes:
                use_dynamic_mcadams = True
            else:
                print(f"Warning: mcadams_coefs_per_frame length ({len(mcadams_coefs_per_frame)}) does not match Nframes ({Nframes}). Falling back to fixed mcadams value.")

        sig_rec = np.zeros([length_sig])

        for m in np.arange(0, Nframes):
            current_mcadams_coef = mcadams_coefs_per_frame[m] if use_dynamic_mcadams else mcadams

            index = np.arange(m * shift, np.minimum(m * shift + winlen, length_sig))
            frame = utterance[index] * win[:len(index)]
            a_lpc = librosa.core.lpc(frame + eps, order=lp_order)
            poles = scipy.signal.tf2zpk(np.array([1]), a_lpc)[1]
            ind_imag = np.where(np.isreal(poles) == False)[0]
            ind_imag_con = ind_imag[np.arange(0, np.size(ind_imag), 2)]

            new_angles = np.angle(poles[ind_imag_con]) ** current_mcadams_coef
            new_angles[np.where(new_angles >= np.pi)] = np.pi
            new_angles[np.where(new_angles <= 0)] = 0

            new_poles = poles.copy()
            for k in np.arange(np.size(ind_imag_con)):
                new_poles[ind_imag_con[k]] = np.abs(poles[ind_imag_con[k]]) * np.exp(1j * new_angles[k])
                new_poles[ind_imag_con[k] + 1] = np.abs(poles[ind_imag_con[k] + 1]) * np.exp(-1j * new_angles[k])

            a_lpc_new = np.real(np.poly(new_poles))
            res = scipy.signal.lfilter(a_lpc, np.array([1]), frame)
            frame_rec = scipy.signal.lfilter(np.array([1]), a_lpc_new, res)
            frame_rec = frame_rec * win[:len(frame_rec)]

            outindex_start = m * shift
            outindex_end = outindex_start + len(frame_rec)
            if outindex_end > length_sig:
                frame_rec = frame_rec[:length_sig - outindex_start]
                outindex = np.arange(outindex_start, length_sig)
            else:
                outindex = np.arange(outindex_start, outindex_end)

            sig_rec[outindex] = sig_rec[outindex] + frame_rec

        max_val = np.max(np.abs(sig_rec))
        if max_val > 0:
            sig_rec = sig_rec / max_val
        else:
            sig_rec = np.zeros_like(sig_rec)

        # Make sure to save as .wav (LibriSpeech is .flac but anonymizer outputs .wav)
        sf.write(output_path, np.float32(sig_rec), sr)

    def do_anonymize(self, dynamic_mcadams=False, min_mcadams_coef=0.7, max_mcadams_coef=0.9, output_base_folder_name="Anon_data"):
        """
        Performs anonymization on all utterances in the DataFrame.
        Saves anonymized files to a new directory structure to prevent overwriting originals.

        Args:
            dynamic_mcadams (bool): If True, use dynamic McAdams coefficients per frame.
                                     Otherwise, use a fixed coefficient (0.8 by default).
            min_mcadams_coef (float): Minimum coefficient for dynamic tuning.
            max_mcadams_coef (float): Maximum coefficient for dynamic tuning.
            output_base_folder_name (str): The name of the new base folder where all anonymized data
                                           will be stored. E.g., "Anon_data". This folder will be
                                           created relative to your 'ASR_Project/data' directory.
        """
        self.speaker_list = self.main_df['speaker_id'].unique().tolist()

        # Determine the project root based on the config file's location.
        # If cfg_path is 'C:/Users/Hans Roozen/Documents/Programming/ASR_Project/config/config.yaml'
        # then os.path.dirname(self.cfg_path) is 'C:/Users/Hans Roozen/Documents/Programming/ASR_Project/config'
        # and os.path.dirname(...) again is 'C:/Users/Hans Roozen/Documents/Programming/ASR_Project'
        project_root = os.path.dirname(os.path.dirname(self.cfg_path))
        
        # Construct the full path to the new base output directory for anonymized files
        anonymized_root_dir = os.path.join(project_root, "data", output_base_folder_name)
        os.makedirs(anonymized_root_dir, exist_ok=True) # Ensure this top-level folder exists

        for speaker_name in tqdm(self.speaker_list, desc="Anonymizing Speakers"):
            selected_speaker_df = self.main_df[self.main_df['speaker_id'] == speaker_name]

            for index, row in selected_speaker_df.iterrows():
                row_relativepath = row['relative_path'] # e.g., 'test-clean/19/198/19-198-0001.flac'
                original_full_path = os.path.join(self.file_path, row_relativepath)

                try:
                    utterance, sr = sf.read(original_full_path)
                except Exception as e:
                    print(f"Error reading {original_full_path}: {e}. Skipping.")
                    continue

                winLengthinms_default = 20
                shiftLengthinms_default = 10
                winlen = np.floor(winLengthinms_default * 0.001 * sr).astype(int)
                shift = np.floor(shiftLengthinms_default * 0.001 * sr).astype(int)
                length_sig = len(utterance)
                Nframes = 1 + np.floor((length_sig - winlen) / shift).astype(int)

                mcadams_coefs_for_frame = None
                
                # Suffix for the output chapter directory name based on anonymization type
                anonymization_type_suffix = ""
                if dynamic_mcadams:
                    mcadams_coefs_for_frame = self._calculate_dynamic_mcadams_coefs(
                        utterance=utterance, sr=sr, winlen=winlen, shift=shift, Nframes=Nframes,
                        min_coef=min_mcadams_coef, max_coef=max_mcadams_coef
                    )
                    anonymization_type_suffix = f"_dynamic_{min_mcadams_coef:.2f}_{max_mcadams_coef:.2f}"
                else:
                    # Generate a random uniform McAdams coefficient for the entire utterance
                    random_mcadams_val = random.uniform(min_mcadams_coef, max_mcadams_coef)
                    # Create an array where each frame uses this random coefficient
                    mcadams_coefs_for_frame = np.full(Nframes, random_mcadams_val)
                    anonymization_type_suffix = f"_random_uniform"

                # Split row_relativepath to get the subset, speaker, chapter, filename
                parts = row_relativepath.split(os.sep)
                if len(parts) < 4:
                    print(f"Warning: Skipping path with unexpected format: {row_relativepath}. Skipping.")
                    continue

                subset_name = parts[0] # e.g., 'test-clean'
                speaker_id = parts[1] # e.g., '19'
                chapter_id = parts[2] # e.g., '198'
                filename_original = parts[3] # e.g., '19-198-0001.flac'
                filename_anonymized = os.path.splitext(filename_original)[0] + '.wav' # Ensure output is .wav

                # Create the new directory path within the anonymized_root_dir
                # Example: ASR_Project/data/Anon_data/test-clean/19/198_fixed_0.80/
                new_chapter_dir = os.path.join(anonymized_root_dir, subset_name, speaker_id, chapter_id + anonymization_type_suffix)
                os.makedirs(new_chapter_dir, exist_ok=True)

                # Construct the full output path for the anonymized file
                output_path = os.path.join(new_chapter_dir, filename_anonymized)
                
                # Perform the anonymization
                # Pass the mcadams_coefs_for_frame, and the 'mcadams' parameter will be overridden
                self.single_anonymize(
                    utterance=utterance,
                    sr=sr,
                    output_path=output_path,
                    mcadams=0.8, # This value will be ignored if mcadams_coefs_per_frame is not None
                    mcadams_coefs_per_frame=mcadams_coefs_for_frame
                )


    def get_mel_preproc(self, x):
        mel_basis_hifi = mel(self.params['preprocessing']['sr'], 1024, fmin=0, fmax=8000, n_mels=80).T
        b, a = self.butter_highpass(30, self.params['preprocessing']['sr'], order=5)
        wav = scipy.signal.filtfilt(b, a, x)
        wav = wav * 0.96 + (np.random.RandomState().rand(wav.shape[0]) - 0.5) * 1e-06
        D = self.pySTFT(wav).T
        D_mel = np.dot(D, mel_basis_hifi)
        mel_spec = np.log(np.clip(D_mel, 1e-5, float('inf'))).astype(np.float32)
        return mel_spec

    def pad_seq(self, x, base=32):
        len_out = int(base * math.ceil(float(x.shape[0]) / base))
        len_pad = len_out - x.shape[0]
        assert len_pad >= 0
        return torch.nn.functional.pad(x, (0, 0, 0, len_pad), value=0), len_pad

    def butter_highpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def pySTFT(self, x, fft_length=1024, hop_length=256):
        x = np.pad(x, int(fft_length // 2), mode='reflect')
        noverlap = fft_length - hop_length
        shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
        strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        fft_window = get_window('hann', fft_length, fftbins=True)
        result = np.fft.rfft(fft_window * result, n=fft_length).T
        return np.abs(result)

    def setup_cuda(self, cuda_device_id=0):
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device('cuda')
        else:
            print("Working on cpu")
            self.device = torch.device('cpu')


class original_dvector_loader:
    def __init__(self, cfg_path='./configs/config.json', M=8):
        """For thresholding and testing.

        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        M: int
            number of utterances per speaker
            must be an even number

        Returns
        -------
        output_tensor: torch tensor
            loaded mel spectrograms
            return shape: (# all speakers, M, embedding size)
        """
        params = read_config(cfg_path)
        self.speaker_list = glob.glob(os.path.join(params['target_dir'], params['dvectors_path_original'], "*.npy"))
        self.M = M


    def provide_test_original(self):
        output_tensor = []

        # return all speakers
        for speaker in self.speaker_list:
            embedding = np.load(speaker)

            # randomly sample a fixed specified length
            if embedding.shape[0] == self.M:
                id = np.array([0])
            elif embedding.shape[0] > self.M:
                id = np.random.randint(0, embedding.shape[0] - self.M, 1)
            else:
                diff = self.M - embedding.shape[0]
                id = np.array([0])
                embedding = np.vstack((embedding, embedding[0:diff]))

            embedding = embedding[id[0]:id[0] + self.M]
            output_tensor.append(embedding)
        output_tensor = np.stack(output_tensor)
        output_tensor = torch.from_numpy(output_tensor)

        return output_tensor



class anonymized_dvector_loader:
    def __init__(self, cfg_path='./configs/config.json', M=8):
        """For d-vector calculation.

        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        M: int
            number utterances per speaker
            must be an even number

        Returns
        -------
        output_tensor: torch tensor
            loaded mel spectrograms
            return shape: (# all speakers, M, embedding size)
        """
        params = read_config(cfg_path)
        self.speaker_list_original = glob.glob(os.path.join(params['target_dir'], params['dvectors_path_original'], "*.npy"))
        self.speaker_list_anonymized = glob.glob(os.path.join(params['target_dir'], params['dvectors_path_anony'], "*.npy"))


        self.M = M
        self.speaker_list_anonymized.sort()
        self.speaker_list_original.sort()
        assert len(self.speaker_list_original) == len(self.speaker_list_anonymized)


    def provide_test_anonymized_and_original(self):
        output_tensor_anonymized = []
        output_tensor_original = []

        # return all speakers of anonymized
        for speaker in self.speaker_list_anonymized:
            embedding = np.load(speaker)

            # randomly sample a fixed specified length
            if embedding.shape[0] == self.M:
                id = np.array([0])
            elif embedding.shape[0] > self.M:
                id = np.random.randint(0, embedding.shape[0] - self.M, 1)
            else:
                diff = self.M - embedding.shape[0]
                id = np.array([0])
                embedding = np.vstack((embedding, embedding[0:diff]))

            embedding = embedding[id[0]:id[0] + self.M]
            output_tensor_anonymized.append(embedding)
        output_tensor_anonymized = np.stack(output_tensor_anonymized)
        output_tensor_anonymized = torch.from_numpy(output_tensor_anonymized)


        # return all speakers of original
        for speaker in self.speaker_list_original:
            embedding = np.load(speaker)

            # randomly sample a fixed specified length
            if embedding.shape[0] == self.M:
                id = np.array([0])
            else:
                id = np.random.randint(0, embedding.shape[0] - self.M, 1)
            embedding = embedding[id[0]:id[0] + self.M]
            output_tensor_original.append(embedding)
        output_tensor_original = np.stack(output_tensor_original)
        output_tensor_original = torch.from_numpy(output_tensor_original)

        return output_tensor_anonymized, output_tensor_original