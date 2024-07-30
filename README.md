# This repository is associated to the article "Identification and Classification of Relativistic Electron Precipitation Events at Earth Using Supervised Deep Learning" by L. Capannolo, W. Li and S. Huang (at Boston University), published in the Frontiers journal. The paper is available at https://www.frontiersin.org/articles/10.3389/fspas.2022.858990/full.

**Please contact me at  luisacap at bu.edu  if you are interested in the code/work.**

<br> <br />
**Abstract**\
We show an application of supervised deep learning in space sciences. We focus on the relativistic electron precipitation into Earth’s atmosphere that occurs when magnetospheric processes (wave-particle interactions or current sheet scattering, CSS) violate adiabatic invariants of trapped radiation belt electrons leading to electron loss. Electron precipitation is a key mechanism of radiation belt loss and can lead to several space weather effects due to its interaction with the Earth’s atmosphere. However, the detailed properties and drivers of electron precipitation are currently not fully understood yet. Here, we aim to build a deep learning model that identifies relativistic precipitation events and their associated driver (waves or CSS). We use a list of precipitation events visually categorized into wave-driven events (REPs, showing spatially isolated precipitation) and CSS-driven events (CSSs, showing an energy-dependent precipitation pattern). We elaborate the ensemble of events to obtain a dataset of randomly stacked events made of a fixed window of data points that includes the precipitation interval. We assign a label to each data point: 0 is for no-events, 1 is for REPs and 2 is for CSSs. Only the data points during the precipitation are labeled as 1 or 2. By adopting a long short-term memory (LSTM) deep learning architecture, we developed a model that acceptably identifies the events and appropriately categorizes them into REPs or CSSs. The advantage of using deep learning for this task is meaningful given that classifying precipitation events by its drivers is rather time-expensive and typically must involve a human. After post-processing, this model is helpful to obtain statistically large datasets of REP and CSS events that will reveal the location and properties of the precipitation driven by these two processes at all L shells and MLT sectors as well as their relative role, thus is useful to improve radiation belt models. Additionally, the datasets of REPs and CSSs can provide a quantification of the energy input into the atmosphere due to relativistic electron precipitation, thus providing valuable information to space weather and atmospheric communities.

<br> <br />
**This repository contains:**

- REPs_classifier_model: folder containing the trained model

- paper_load_model_poes_predictions.py: code to load the trained model and a full day of POES/MetOp data and provide the CSS and REP events identified by the model on the selected POES/MetOp date

- paper_library.py: routines used in paper_load_model_poes_predictions.py

- LSTM_scaler.joblib: normalization factors of the training dataset
- See 2024 UPDATES below to use post-processing

<br> <br />
**Additional Information**

- POES/MetOp data are available at: https://www.ncei.noaa.gov/data/poes-metop-space-environment-monitor/access/l1b/v01r00/ ~~https://satdat.ngdc.noaa.gov/sem/poes/data/processed/ngdc/uncorrected/full/~~

- The dataset preparation and model training are done on a Linux OS (version 3.10.0-1160.49.1.el7.x86_64) machine (Shared Computer Cluster at Boston University) in Python (version 3.8.6), using the TensorFlow library (version 2.5.0, https://www.tensorflow.org) and the Python packages: Matplotlib (https://matplotlib.org), Scikit Learn (https://scikit-learn.org/stable/), Xarray (https://xarray.pydata.org/en/stable/), Joblib (https://joblib.readthedocs.io/en/latest/), Seaborn (https://seaborn.pydata.org/), Numpy (https://numpy.org), and Pandas (https://pandas.pydata.org)

<br> <br />
**Funding:** This research is supported by the NSF grants AGS-1723588 and AGS-2019950, the NASA grants 80NSSC20K0698, 80NSSC21K1385, and 80NSSC20K1270, and the Alfred P. Sloan Research Fellowship FG-2018-10936.

<br> <br />
**Blog post**: https://blogs.bu.edu/luisacap/2024/02/11/applying-lstm-to-a-multi-class-classification-problem/

<br> <br />
**July 2024 UPDATES**: post-processing has been added in paper_library_2023update.py. To load the trained model, with post-processing, use paper_load_model_poes_predictions_2023update.py
