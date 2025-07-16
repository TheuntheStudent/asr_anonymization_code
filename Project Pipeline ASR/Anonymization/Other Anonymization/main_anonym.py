from inference_speaker import mcadams_anonymization_process_e2e

# Run the anonymization
# mcadams_anonymization_process_e2e(
#     global_config_path="C:/Users/Hans Roozen/Documents/Programming/ASR_Project/config/config.yaml",
#     output_utter_dirname='Anon_data_McAdams_Random',
#     mcadams_coef=0.80
# )

mcadams_anonymization_process_e2e(global_config_path="C:/Users/Hans Roozen/Documents/Programming/ASR_Project/config/config.yaml",
                                      dynamic_mcadams=False, min_mcadams_coef=0.7, max_mcadams_coef=0.9,
                                      output_base_folder_name='Anon_data_McAdams_Random')



