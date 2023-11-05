[![build](https://github.com/ramp-kits/bovine_embryo_survival_prediction/actions/workflows/testing.yml/badge.svg)](https://github.com/ramp-kits/bovine_embryo_survival_prediction/actions/workflows/testing.yml)
# Bovine embryos survival prediction data Challenge

This is my contribution to the kaggle challenge of the IPP Data Science master (2022-2023) 

This repository contains :
- Starting kit for the `bovine_embryo_survival_prediction` Challenge
- My code to train and test various deep learning models under the `src` folder

# Setup
Run the followings commands :
- Clone the repository 
- Create the environment `conda env create -f environment.yml` (install pytorch by yourself if you want to use your cuda)
- Activate the environment `conda activate bovine_embryo_survival_prediction`
- Download the public data via the dedicated python script `python download_data.py`
- Have a look this jupyter notebook to understand the challenge : `bovine_embryo_survival_prediction_starting_kit.ipynb`


