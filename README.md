## Project Introduction

**PIMOD** (Population to Individual Mood Decoder) is an open-source toolkit designed for processing, analyzing, and modeling intracranial EEG (iEEG) data, with a primary focus on decoding mood states from neural signals in real time. Code repository accompanying the manuscript"PIMOD: A Two-Step Deep-Learning Framework for Real-Time Mood Decoding and Personalized Neurostimulation Response Prediction". For technical inquiries, feedback, or collaboration requests, please contact us via email at Mic2462533841@outlook.com.

## Project Structure

This section outlines the organization of the repository and the purpose of its directories and files to help you navigate and utilize this project effectively.

### Repository Layout

```plaintext
PIMOD/
│
├── config.py
# Configuration parameters and command-line argument parsing.
│
├── dataloader.py
# Data loading and preprocessing.
│
├── data_processing.py
# Processing the raw dataset.
│
├── function.py
# Label processing, normalization.
│
├── main.py
# The main script for model training, fine-tuning, and evaluation.
│
├── model.py
# Model definitions.
│
├── solver.py
# Core logic for training, fine-tuning, testing, and result saving.
│
├── requirements.txt
# Python dependencies list.
│
└── README.md
```


## Before You Start

Before running any experiments, please complete the following steps to set up your environment and configure the project:

###  Environment setting

Install all required dependencies using the provided `requirements.txt` file.  
Open a terminal in the project directory and run:

```
pip install -r requirements.txt
```

This will ensure all necessary Python packages are installed.

### Configure Data Paths and Parameters

- **Edit and run `data_processing.py`:**  
  Update the file to match your raw data paths and settings, then execute it to preprocess the initial dataset.

- **Edit `dataloader.py`:**  
  Set the correct data paths for your iEEG data by modifying the `path` and `stim_data_path` variables at the top of the file.

- **Edit `config.py`:**  
  Adjust configuration parameters (such as model type, training settings, and result paths) as needed. You can modify default values in the `Config` class or use command-line arguments when running the main script.

Make sure these settings match your local data and experiment requirements before proceeding.
