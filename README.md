# REPs_classifier_codes_for_paper
REPs_classifier_model: folder containing the trained model
paper_load_model_poes_predictions.py: code to load the trained model and a full day of POES/MetOp data and provide the CSS and REP events identified by the model on the selected POES/MetOp date
paper_library.py: routines used in paper_load_model_poes_predictions.py
LSTM_scaler.joblib: normalization factors of the training dataset
POES/MetOp data are available at: https://satdat.ngdc.noaa.gov/sem/poes/data/processed/ngdc/uncorrected/full/
