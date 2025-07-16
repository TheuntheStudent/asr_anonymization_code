"""
Created on November 10, 2019
functions for writing/reading data to/from disk

@modified_by: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
"""
import yaml
import numpy as np
import os
import shutil
import pdb




def read_config(config_path):
    """Reads config file in yaml format into a dictionary

    Parameters
    ----------
    config_path: str
        Path to the config file in yaml format

    Returns
    -------
    config dictionary
    """
    print(config_path)
    with open(config_path, 'rb') as yaml_file:
        return yaml.safe_load(yaml_file)


def write_config(params, cfg_path, sort_keys=False):
    with open(cfg_path, 'w') as f:
        yaml.dump(params, f)


def create_experiment(experiment_name, global_config_path):
    params = read_config(global_config_path)
    params['experiment_name'] = experiment_name
    create_experiment_folders(params)
    cfg_file_name = params['experiment_name'] + '_config.yaml'
    cfg_path = os.path.join(os.path.join(params['target_dir'], params['network_output_path']), cfg_file_name)
    params['cfg_path'] = cfg_path
    write_config(params, cfg_path)
    return params


def create_experiment_folders(params):
    try:
        path_keynames = ["network_output_path", "tb_logs_path", "stat_log_path", "output_data_path", "dvectors_path", "dvectors_foranonym_path", "dvectors_path_anonymized", "dvectors_path_original",
                         "dvectors_path_anony_dysarthria", "dvectors_path_anony_dysglossia", "dvectors_path_anony_dysphonia",
                         "dvectors_path_original_dysarthria", "dvectors_path_original_dysglossia", "dvectors_path_original_dysphonia"]
        for key in path_keynames:
            params[key] = os.path.join(params['experiment_name'], params[key])
            os.makedirs(os.path.join(params['target_dir'], params[key]))
    except:
        raise Exception("Experiment already exist. Please try a different experiment name")


# def open_experiment(experiment_name, global_config_path):
    # """Open Existing Experiments
    # """
    # default_params = read_config(global_config_path)
    # cfg_file_name = experiment_name + '_config.yaml'
    # cfg_path = os.path.join(os.path.join(default_params['target_dir'], experiment_name, default_params['network_output_path']), cfg_file_name)
    # params = read_config(cfg_path)
    # return params
def open_experiment(experiment_name, global_config_path):
    """
    Parameters
    ----------
    experiment_name: str
        Name of the experiment (e.g., 'baseline_speaker_model').
        This name will be used to create a subdirectory within the main output folder
        to organize results for this specific run.
    global_config_path: str
        Path to the main global configuration file (e.g., 'config/config.yaml').

    Returns
    -------
    dict: A dictionary containing all parameters from the global config,
          plus 'target_dir' (the path to the experiment's output directory)
          and 'cfg_path' (which is the global_config_path itself, used by other modules).
    """
    # Read the main global configuration file
    params = read_config(global_config_path)

    # Determine the project root (e.g., 'C:/Users/Hans Roozen/Documents/Programming/ASR_Project')
    # This assumes global_config_path is structured like 'PROJECT_ROOT/config/config.yaml'
    project_root = os.path.dirname(os.path.dirname(global_config_path))

    # Define the base directory where all experiment outputs will be stored.
    # You can configure 'output_base_dir' in your config.yaml (e.g., 'output' or 'anonym-AudioWAV-2').
    # If not specified in config, it defaults to 'output'.
    output_base_dir_name = params.get('output_base_dir', 'output')
    
    # Construct the full path to the specific experiment's output directory
    # e.g., 'PROJECT_ROOT/output/baseline_speaker_model'
    target_dir = os.path.join(project_root, output_base_dir_name, experiment_name)
    
    # Create the target directory if it doesn't already exist
    # This is generally a good practice to ensure paths exist before writing files.
    os.makedirs(target_dir, exist_ok=True)

    # Add the calculated target_dir and the global_config_path itself to the parameters dictionary.
    # The 'target_dir' will be used to save d-vectors, models, logs, etc.
    # The 'cfg_path' is needed by other modules (like Prediction) which re-read the config.
    params['target_dir'] = target_dir
    params['cfg_path'] = global_config_path

    return params    
    


def delete_experiment(experiment_name, global_config_path):
    """Delete Existing Experiment folder
    """
    default_params = read_config(global_config_path)
    cfg_file_name = experiment_name + '_config.yaml'
    cfg_path = os.path.join(os.path.join(default_params['target_dir'], experiment_name, default_params['network_output_path']), cfg_file_name)
    params = read_config(cfg_path)
    shutil.rmtree(os.path.join(params['target_dir'], experiment_name))
