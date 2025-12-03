Project Summary

Title: Customer Churn Prediction using ANN
One-line: Developed an end-to-end Artificial Neural Network pipeline to predict customer churn from banking data, including preprocessing, feature encoding, model training, hyperparameter tuning, and model export.
Problem Statement

Problem: Build a predictive model to identify customers likely to churn (binary classification) using the provided Churn_Modelling.csv.
Goal: Improve retention by flagging customers at high risk of leaving so the business can take targeted actions.
Approach & Model

Data preparation: Removed irrelevant identifiers, encoded categorical features (Gender with LabelEncoder, Geography with OneHotEncoder), and scaled numeric features using StandardScaler. Saved encoders and scaler as pickles for reproducible inference.
Model: Built and trained an ANN (TensorFlow / Keras) with two hidden layers (examples used: 64 and 32 units, ReLU activations) and a sigmoid output for binary classification. Used EarlyStopping and TensorBoard for training monitoring and to prevent overfitting.
Hyperparameter tuning: Explored network depth and width with scikeras + GridSearchCV (neurons, layers, epochs) to find better-performing architectures.
Artifacts: Saved final model as model.h5 and preprocessing objects as scaler.pkl, label_encoder_gender.pkl, onehot_encoder_geo.pkl for later inference.
Tech Stack

Languages / Libraries: Python, NumPy, pandas, scikit-learn, TensorFlow / Keras, scikeras, GridSearchCV
Utilities: pickle, TensorBoard
Files in repo: Churn_Modelling.csv, experiments.ipynb, hyperparametertuningann.ipynb, prediction.ipynb, model.h5, scaler.pkl, encoder pickles
Dev / runtime: Jupyter Notebooks; requirements listed in requirements.txt
Suggested Resume Bullets

Project: Customer Churn Prediction (ANN) — Built an end-to-end machine learning pipeline to predict customer churn using a real-world banking dataset (Churn_Modelling.csv).
Data & Preprocessing: Implemented data cleaning, label encoding and one-hot encoding for categorical features, and standardized numeric features; serialized preprocessing objects for consistent production inference.
Modeling: Designed and trained a Keras-based ANN with early stopping and TensorBoard monitoring; exported the trained model as model.h5.
Hyperparameter Tuning: Tuned network architecture (layers, neurons, epochs) using scikeras with GridSearchCV to optimize performance.
Deployment Readiness: Packaged preprocessing artifacts (scaler.pkl, encoder pickles) alongside the saved model to enable reproducible predictions in the prediction.ipynb.
Tools: Python, pandas, scikit-learn, TensorFlow/Keras, scikeras, GridSearchCV, TensorBoard, pickle
