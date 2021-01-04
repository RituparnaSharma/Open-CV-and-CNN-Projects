import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


data = pd.read_csv('./data/Real_Combine.csv')
print(data.head())

Independent_Features = data.iloc[:, :-1]
dependent_Feature = data.iloc[:, -1]


def Model(hp):
    model = keras.Sequential()
    for item in range(hp.Int('num_layers', 2, 20)):
        model.add(layers.Dense(units=hp.Int('units_' + str(item),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss="mean_absolute_error",
        metrics=['mean_absolute_error'])
    return model


tuner = RandomSearch(
    Model,
    objective='val_mean_absolute_error',
    max_trials=5,
    executions_per_trial=3,
    directory='tuned parameters',
    project_name='Air quality Index')

X_train, X_test, y_train, y_test = train_test_split(Independent_Features, dependent_Feature, test_size=0.2)

tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

print(tuner.results_summary())

print(tuner.tuner.get_best_models(num_models=2))
