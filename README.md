# TIML: Temporal-Incremental Malware Learning

Welcome to the TIML project. This repository contains the codebase and necessary information to replicate the experiments in our paper on Temporal-Incremental Malware Learning.

## Environment Setup

1. **Install Required Packages:**
   Use the provided `environment.yaml` file to create a new Conda environment and install the required packages.
    ```bash
    conda env create --file environment.yaml
    ```

2. **Activate the TIML Environment:**
   Switch to the newly created environment using the following command:
    ```bash
    conda activate TIML
    ```

## Dataset Preparation

1. **Download MalNet Data:**
   Obtain MalNet images and labels from the [MalNet Website](https://mal-net.org/).

2. **Download APKs Using Malware Hash:**
   Utilize the malware hash corresponding to each image in the MalNet dataset to download the original APK from [AndroZoo](https://androzoo.uni.lu/).

3. **Generate MalScan Features:**
   Use the [MalScan Tool](https://github.com/Trustworthy-Software/Reproduction-of-Android-Malware-detection-approaches/tree/master#malscan) to process the APKs and extract MalScan features.

## Data Preprocessing

1. **Generate Dataset Statistics:**
   Execute the `chronological_statistics.py` script to compute essential statistics of the dataset.
    ```bash
    python chronological_statistics.py
    ```

2. **Restructure the Dataset:**
   Run the `data_preprocess` script to organize the dataset according to the TIML paradigm.
    ```bash
    python data_preprocess.py
    ```

## Training and Evaluation

1. **Set Experiment Configuration:**
   Modify the configuration files within the `exp_settings` folder to suit each approach you intend to test.

2. **Train and Evaluate the Model:**
   Execute experiments based on the selected configurations. Examples are provided below:
    - For MalNet image:
        ```bash
        python main.py --exp_setting exp_settings/full_malnet_data/upper_bound.yaml
        ```
    - For MalScan features:
        ```bash
        python main_malscan.py --exp_setting exp_settings/full_malscan_data/lwf_w_exemplar_malscan.yaml
        ```
