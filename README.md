# cs2565-project

Authors Lucas Martinez, Alex Peterson, Tapan Khaladkar

# Installation
pipenv sync
pipenv shell
python -m ipykernel install --user --name=cs2565-env
jupyter notebook

# Notebook Descriptions

## Feature Extraction 
Code to featurize text data from linkedin dataset:
cleaned_notebooks/linkedin_feature_extraction.ipnyb

Code to featurize text data from scotus dataset:
cleaned_notebooks/scotus_feature_extraction.ipnyb

## Regressor Training
Code to train regressors for linkedin dataset:
cleaned_notebooks/train_regressor_linkedin.ipnyb

Code to train regressors for scotus dataset:
cleaned_notebooks/train_regressor_scotus.ipnyb

## Doc2Vec and SBERT
Note: 
Featurizing and regressor training for Doc2Vec and SBERT were done in a single notebook each.

Linkedin experiments can be found in:
experiments/linkedin_models/Doc2Vec.ipynb
experiments/linkedin_models/SBERT.ipynb

Scotus experiments can be found in:
experiments/scotus_models/Doc2Vec.ipynb
experiments/scotus_models/SBERT.ipynb

## Results
results directory contains results for GBT Regressor models for all featurizing strategies for both linkedin and scotus datasets. Each directory includes a pickle file of raw score dictionary (mae, rmse, r2_score), a pickled model object (sklearn.ensemble.GradientBoostingRegressor) and a jpeg graph of the residuals. 

### Unpickling with joblib

Please install sklearn==1.5.2
```
!pip install sklearn==1.5.2
```

Load pickled models:
```
import joblib
from sklearn.ensemble import GradientBoostingRegressor

# Load the object from the file
loaded_model = joblib.load('results/linkedin/bigram/model_object.pkl') 
loaded_scores = joblib.load('results/linkedin/bigram/model_scores.pkl') 

```