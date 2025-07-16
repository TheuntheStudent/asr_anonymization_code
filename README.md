# The Impact of Speech Anonymization on Pathology and Its Limits



Overview
------

* This is a repository for our project for the course (Automatic) Speech Recognition. This repository is based on the repository of the paper [**The Impact of Speech Anonymization on Pathology and Its Limits**](https://arxiv.org/abs/2404.08064). (Tayebi et al., 2024)


### Prerequisites

The software is developed in **Python 3.9**. 

Main Python modules required for the software can be installed from ./requirements:

```
$ conda env create -f requirements.yaml
$ conda activate pathology_anonym
```

Code structure
---

The code for the different experiments is available here.

1. To anonymise speech, go to the folder *./Project Pipeline ASR/Anonymization/*
* The data preprocessing parameters, directories, hyper-parameters, and model parameters can be modified from *./configs/config.yaml*.
* *./Default Anonymization/* directory contains all the files needed for anonymization using the fixed McAdams coefficient method. 
* *./Other Anonymization/* directory contains all the files needed for anonymization using the Random and Dynamic McAdams coefficient method.
* Within these folders the *./inference_speaker_data_loader.py* contains the code for the actual anonymization process. To run the Random McAdams method  (in *./Other Anonymization/inference_speaker_data_loader.py*) change the **dynamic_mcadams** parameter in *./main_anonym.py* to **False** and for the Dynamic approach to **True**.
* Equal Error Rate and Word Error Rate can be computed using the *./eer_calc.py* and *./wer_calc.py* files.

2. To perform the emotion classification, go to folder *./Project Pipeline ASR/Emotion classification/*
* The notebook contains the full pipeline, including feature extraction, classification, and evaluation. The notebook requires access to:
    * Separate zipped folders of the audio files provided in the AudioWAV folder of https://github.com/CheyneyComputerScience/CREMA-D and their anonymized version
    * An CSV file with the relative_path and speaker_id of each audio file 
    * Optional: A checkpoint file pointing to the best performing model weights (if not wanting to retrain, but requires small code adjustment)
* If these are provided, the notebook can be ran to perform the entire emotion classification pipeline