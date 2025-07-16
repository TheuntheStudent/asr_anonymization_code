from inference_speaker import mcadams_anonymization_process_e2e

# Run the anonymization
mcadams_anonymization_process_e2e(
    global_config_path="config/config.yaml",
    output_utter_dirname="C:/Users/theun/ASR_Github/PathologyAnonym/mcAdams_Anonym/anonym-AudioWAV-2",
    mcadams_coef=0.75
)


