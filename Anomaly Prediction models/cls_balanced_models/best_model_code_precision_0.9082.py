import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.builtins import ZeroCount
from xgboost import XGBClassifier

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=23)

# Average CV score on the training set was:0.908185740236149
exported_pipeline = make_pipeline(
    ZeroCount(),
    XGBClassifier(learning_rate=0.1, max_depth=7, min_child_weight=6, n_estimators=100, nthread=1, subsample=0.7000000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
