import numpy as np
import pandas as pd
import pickle
import streamlit
# Load the saved vectorizer and model
feature_extraction = pickle.load(open('E:/feature_extraction.sav', 'rb'))
loaded_model = pickle.load(open('E:/trained_model.sav', 'rb'))