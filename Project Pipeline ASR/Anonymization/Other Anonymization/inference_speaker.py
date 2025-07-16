"""
inference_speaker.py
Created on Oct 30, 2023.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""
import pdb
import os
import warnings

# Correct import for open_experiment and read_config
from config.serde import open_experiment, read_config # Assuming read_config is also in serde.py
# Make sure this import path is correct for your project structure
from inference_speaker_data_loader import loader_for_dvector_creation, anonymizer_loader
from speaker_Prediction import Prediction # This is your Prediction class
from models.lstm import SpeechEmbedder # This is your SpeechEmbedder model

warnings.filterwarnings('ignore')


# Updated function signature and call to anonymizer_loader.do_anonymize
def mcadams_anonymization_process_e2e(global_config_path="C:/Users/Hans Roozen/Documents/Programming/ASR_Project/config/config.yaml",
                                      dynamic_mcadams=False, min_mcadams_coef=0.7, max_mcadams_coef=0.9,
                                      output_base_folder_name='Anon_data_McAdams_Random'): # Renamed output_utter_dirname for clarity
    """
    Orchestrates the McAdams anonymization process.

    Parameters
    ----------
    global_config_path: str
        Path to the global config.yaml file.
    dynamic_mcadams (bool): If True, use dynamic McAdams coefficients per frame.
                            Otherwise, use a fixed coefficient (0.8 by default).
    min_mcadams_coef (float): Minimum coefficient for dynamic tuning.
    max_mcadams_coef (float): Maximum coefficient for dynamic tuning.
    output_base_folder_name (str): The name of the new base folder where all anonymized data
                                   will be stored. E.g., "Anon_data_McAdams_Fixed".
    """
    print('\nAnonymizing all utterances of each speaker....')
    print('Loop over speakers....')
    data_handler_anonymizer = anonymizer_loader(cfg_path=global_config_path, nmels=40)
    
    # Call do_anonymize with the correct parameters
    data_handler_anonymizer.do_anonymize(
        dynamic_mcadams=dynamic_mcadams,
        min_mcadams_coef=min_mcadams_coef,
        max_mcadams_coef=max_mcadams_coef,
        output_base_folder_name=output_base_folder_name
    )
    
    print('Anonymization done!')


# Updated function signature and removed filtering logic
def anonymized_EER_calculation_e2e(global_config_path="C:/Users/Hans Roozen/Documents/Programming/ASR_Project/config/config.yaml",
                                    experiment_name='baseline_speaker_model', epochs=1000, M=8, spk_nmels=40,
                                    anonym_base_folder_name='Anon_data_McAdams', # New parameter
                                    anonymization_type_suffix='_fixed_0.80'): # New parameter
    """
    Calculates EER for anonymized signals by creating d-vectors from the anonymized audio.

    Parameters
    ----------
    global_config_path: str
        Path to the global config.yaml file.
    experiment_name: str
        Name of the experiment.
    epochs: int
        Total number of epochs to do the evaluation process.
        The results will be the average over the result of each epoch.
    M: int
        Number of utterances per speaker for enrollment/verification (M/2 each).
    spk_nmels: int
        Number of mel-frequency bins for d-vector creation.
    anonym_base_folder_name (str): The top-level folder name where anonymized data is stored.
    anonymization_type_suffix (str): The suffix appended to chapter folders during anonymization.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']

    # d-vector and EER calculation
    predictor = Prediction(cfg_path)
    model = SpeechEmbedder(nmels=params['preprocessing']['nmels'], hidden_dim=params['Network']['hidden_dim'],
                           output_dim=params['Network']['output_dim'], num_layers=params['Network']['num_layers'])

    predictor.setup_model_for_inference(model=model)
    print('Preprocessing for d-vector creation....')
    data_handler = loader_for_dvector_creation(cfg_path=cfg_path, spk_nmels=spk_nmels)

    # REMOVED ALL FILTERING LOGIC FOR subset, mic_room, patient_control
    # data_handler.main_df = data_handler.main_df[data_handler.main_df['subset'] == subset]
    # if subsetname == 'dysphonia':
    #     data_handler.main_df = data_handler.main_df[data_handler.main_df['mic_room'] == 'logitech']
    # ... (and so on)
    # data_handler.main_df = data_handler.main_df[data_handler.main_df['patient_control'] == 'patient']

    # Pass the correct folder name and suffix to provide_data_anonymized
    data_loader = data_handler.provide_data_anonymized(
        anonym_base_folder_name=anonym_base_folder_name,
        anonymization_type_suffix=anonymization_type_suffix
    )
    print('\nPreprocessing done!')

    print('Creating the d-vectors (network prediction) for the anonymized signals....')
    # Removed subsetname parameter from dvector_prediction call
    predictor.dvector_prediction(data_loader, anonymized=True)

    print('\nEER calculation....')
    # Removed subsetname parameter from EER_newmethod_epochy_anonymized call
    avg_EER_test, std_EER, numspk = predictor.EER_newmethod_epochy_anonymized(cfg_path, M=M, epochs=epochs)

    print('\n----------------------------------------------------------------------------------------')
    print(f'Speaker model: GE2E trained on Librispeech | speaker model No. mels: {int(spk_nmels)}\n '
          f'No. enrolment/verification utterances per speaker: {int(M/2)}/{int(M/2)} | No. speakers: {int(numspk)}')
    print(f'\n\tAverage EER over {epochs} repetitions: {(avg_EER_test) * 100:.2f} ± {(std_EER) * 100:.2f}%')

    # saving the stats
    mesg = (f'\n----------------------------------------------------------------------------------------\n'
            f"Speaker model: GE2E trained on Librispeech | speaker model No. mels: {int(spk_nmels)}\n"
            f"No. enrolment/verification utterances per speaker: {int(M/2)}/{int(M/2)} | No. speakers: {int(numspk)}\n"
            f"\n\tAverage EER over {epochs} repetitions: {(avg_EER_test) * 100:.2f} ± {(std_EER) * 100:.2f}%\n"
            f'\n----------------------------------------------------------------------------------------\n')
    
    log_dir = os.path.join(params['target_dir'], params['stat_log_path'])
    os.makedirs(log_dir, exist_ok=True)
    
    # Log file name simplified to remove subsetname and include anonymization suffix
    log_file_name = f'test_results_anonymized_M{M}_{anonymization_type_suffix.replace("_", "")}.txt' 
    with open(os.path.join(log_dir, log_file_name), 'a') as f:
        f.write(mesg)


# Updated function signature and removed filtering logic
def direct_clssical_EER_calculation_e2e(global_config_path="C:/Users/Hans Roozen/Documents/Programming/ASR_Project/config/config.yaml",
                                         experiment_name='baseline_speaker_model', epochs=1000, M=8, spk_nmels=40): # Removed subsetname, subset
    """Main function for creating d-vectors & testing for original signals.
    Purpose here is validation of the model

    Parameters
    ----------
    global_config_path: str
        Path to the global config.yaml file.
    experiment_name: str
        Name of the experiment.
    epochs: int
        Total number of epochs to do the evaluation process.
        The results will be the average over the result of each epoch.
    M: int
        Number of utterances per speaker.
    spk_nmels: int
        Number of mel-frequency bins.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    predictor = Prediction(cfg_path)
    model = SpeechEmbedder(nmels=params['preprocessing']['nmels'], hidden_dim=params['Network']['hidden_dim'],
                           output_dim=params['Network']['output_dim'], num_layers=params['Network']['num_layers'])

    predictor.setup_model_for_inference(model=model)

    print('Preprocessing for d-vector creation....')
    data_handler = loader_for_dvector_creation(cfg_path=cfg_path, spk_nmels=spk_nmels)

    # REMOVED ALL FILTERING LOGIC FOR subset, mic_room, patient_control

    data_loader = data_handler.provide_data_original()
    print('\nPreprocessing done!')

    print('Creating the d-vectors (network prediction)....')
    # Removed subsetname parameter from dvector_prediction call
    predictor.dvector_prediction(data_loader, anonymized=False)

    print('\nEER calculation....')
    # Removed subsetname parameter from EER_newmethod_epochy call
    avg_EER_test, std_EER, numspk = predictor.EER_newmethod_epochy(cfg_path, M=M, epochs=epochs)

    print('\n----------------------------------------------------------------------------------------')
    print(f'Speaker model: GE2E trained on Librispeech | speaker model No. mels: {int(spk_nmels)}\n '
          f'No. enrolment/verification utterances per speaker: {int(M/2)}/{int(M/2)} | No. speakers: {int(numspk)}')
    print(f'\n\tAverage EER over {epochs} repetitions: {(avg_EER_test) * 100:.2f} ± {(std_EER) * 100:.2f}%')

    # saving the stats
    mesg = (f'\n----------------------------------------------------------------------------------------\n'
            f"Speaker model: GE2E trained on Librispeech | speaker model No. mels: {int(spk_nmels)}\n"
            f"No. enrolment/verification utterances per speaker: {int(M/2)}/{int(M/2)} | No. speakers: {int(numspk)}\n"
            f"\n\tAverage EER over {epochs} repetitions: {(avg_EER_test) * 100:.2f} ± {(std_EER) * 100:.2f}%\n"
            f'\n----------------------------------------------------------------------------------------\n')
    
    log_dir = os.path.join(params['target_dir'], params['stat_log_path'])
    os.makedirs(log_dir, exist_ok=True)
    
    # Log file name simplified
    with open(os.path.join(log_dir, f'test_results_original_M{M}.txt'), 'a') as f:
        f.write(mesg)


if __name__ == '__main__':
    # Define your global config path
    GLOBAL_CONFIG_PATH = "C:/Users/Hans Roozen/Documents/Programming/ASR_Project/config/config.yaml"

    # # --- Step 1: Run Fixed Coefficient Anonymization ---
    # print("\n--- Running Fixed Coefficient Anonymization ---")
    # mcadams_anonymization_process_e2e(
    #     global_config_path=GLOBAL_CONFIG_PATH,
    #     dynamic_mcadams=False,
    #     output_base_folder_name='Anon_data_McAdams_Fixed' # This will create folders like chapter_id_fixed_0.80
    # )
    
    # # --- Step 2: Run Dynamic Coefficient Anonymization ---
    # print("\n--- Running Dynamic Coefficient Anonymization ---")
    # mcadams_anonymization_process_e2e(
    #     global_config_path=GLOBAL_CONFIG_PATH,
    #     dynamic_mcadams=True,
    #     min_mcadams_coef=0.7,
    #     max_mcadams_coef=0.9,
    #     output_base_folder_name='Anon_data_McAdams_Dynamic_0.70_0.90' # This will create folders like chapter_id_dynamic_0.70_0.90
    # )

    print("\n" + "="*80 + "\n")

    # --- Step 3: Evaluate Original Audio EER ---
    print("\n--- Evaluating Original Audio EER ---")
    direct_clssical_EER_calculation_e2e(
        global_config_path=GLOBAL_CONFIG_PATH,
        experiment_name='baseline_speaker_model',
        epochs=100, # Use a reasonable number of epochs for robust EER
        M=8,
        spk_nmels=40
    )

    print("\n" + "="*80 + "\n")

    # --- Step 4: Evaluate Fixed Coefficient Anonymized Audio EER ---
    print("\n--- Evaluating Fixed Coefficient Anonymized Audio EER ---")
    anonymized_EER_calculation_e2e(
        global_config_path=GLOBAL_CONFIG_PATH,
        experiment_name='baseline_speaker_model',
        epochs=100,
        M=8,
        spk_nmels=40,
        anonym_base_folder_name='Anon_data_McAdams_Fixed',
        anonymization_type_suffix='_fixed_0.80' # Must match the suffix used during anonymization
    )

    print("\n" + "="*80 + "\n")

    # --- Step 5: Evaluate Dynamic Coefficient Anonymized Audio EER ---
    print("\n--- Evaluating Dynamic Coefficient Anonymized Audio EER ---")
    anonymized_EER_calculation_e2e(
        global_config_path=GLOBAL_CONFIG_PATH,
        experiment_name='baseline_speaker_model',
        epochs=100,
        M=8,
        spk_nmels=40,
        anonym_base_folder_name='Anon_data_McAdams_Dynamic_0.70_0.90',
        anonymization_type_suffix='_dynamic_0.70_0.90' # Must match the suffix used during anonymization
    )

    print("\nAll EER evaluations complete.")