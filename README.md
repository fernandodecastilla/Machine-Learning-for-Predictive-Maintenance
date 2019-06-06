# Machine-Learning-for-Predictive-Maintenance
Anomaly detection and failure prognosis applied to industrial machines

The design and implementation of the Master's Final Project is structured in three directories:

Notebooks: implementation of the MFP in Python through phases of the life cycle.
Anomaly Detection models: anomaly detection model.
Anomaly Prediction models: classification models for prediction of anomalies.
Within the 'Notebooks' directory:

Directory 00: Notebook to execute and obtain an interactive Exploratory Data Analysis, in which the user has a series of widgets at his disposal to configure. It is not necessary to run this Notebook to continue with the rest.
Directory 01: Initial notebook with an optimal loading of the starting dataset, preparation of data, generation of new variables (feature engineering), and reduction of dimensionality through AutoEncoders.
Directory 02: Notebook with a second phase of feature engineering, focused on operations to obtain mobile statistical measures, adapting the code to deal with Memory Error messages.
Directory 03: Generation of a balanced dataset between anomalous and non-anomalous records, design of an anomaly detection model by means of an AutoEncoder, and analysis of its suitability from the business point of view. To give credibility to an initial model of anomalies detection, it is key to minimize the number of false positives (false anomalies), at the expense of increasing the number of false negatives (anomalies not reported).
Directory 04: Generation of datasets to apply a battery of classification models on categorical features.
Directory 05: Generation of datasets to apply a battery of classification models on categorical and numerical features.
Directory 06: Generation of datasets to apply two MLP models (Multi Layer Perceptron). The first on all available features, and the second on only those features whose values ​​reach a minimum of standard deviation.
