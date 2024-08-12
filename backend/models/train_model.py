# models/model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import holidays
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import HyperModel, RandomSearch
import h5py
import joblib

merged_data = pd.read_csv('../data/merged_data.csv')


# Function to create dataset for LSTM with multiple time horizons
def create_dataset(dataset, look_back=12, horizons=[1, 2, 3, 4, 8, 12]):
    X, Y = [], []
    for i in range(len(dataset) - max(horizons) - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append([dataset[i + h, 1] for h in horizons])  # Multiple horizons
    return np.array(X), np.array(Y)

# Function to calculate accuracy based on a threshold
def calculate_accuracy(y_true, y_pred, threshold=0.1):
    return np.mean(np.abs(y_true - y_pred) <= threshold)

# Extract relevant features for LSTM
features = ['timestamp', 'congestion_level', 'temperature', 'precipitation', 
            'wind_speed', 'visibility', 'humidity', 'pressure', 'hour', 'day_of_week']
data = merged_data[features].copy()

# Feature engineering
data['day_of_month'] = pd.to_datetime(data['timestamp']).dt.day
data['month'] = pd.to_datetime(data['timestamp']).dt.month
nigeria_holidays = holidays.Nigeria()
data['is_holiday'] = pd.to_datetime(data['timestamp']).dt.date.apply(lambda x: 1 if x in nigeria_holidays else 0)

# Convert timestamp to datetime index
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Normalize the dataset
scalers = {}
for column in data.columns:
    scalers[column] = MinMaxScaler(feature_range=(0, 1))
    data[column] = scalers[column].fit_transform(data[column].values.reshape(-1, 1))

# Set a single look-back value and forecast horizons
look_back = 12  # You can change this value to experiment with different look-back periods
horizons = [1, 2, 3, 4, 8, 12]  # 15 mins, 30 mins, 45 mins, 1 hour, 2 hours, and 3 hours (assuming data is in 15-minute intervals)

# Split into train and test sets (80% train, 20% test)
train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
train, test = data.iloc[0:train_size, :], data.iloc[train_size:len(data), :]

# Create dataset for LSTM
X_train, Y_train = create_dataset(train.values, look_back, horizons)
X_test, Y_test = create_dataset(test.values, look_back, horizons)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Define the HyperModel class for Keras Tuner
class TrafficPredictionHyperModel(HyperModel):
    def __init__(self, input_shape, num_outputs):
        self.input_shape = input_shape
        self.num_outputs = num_outputs
    
    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(Bidirectional(LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32), return_sequences=True)))
        model.add(Dropout(hp.Float('dropout_1', 0.1, 0.5, step=0.1)))
        model.add(Bidirectional(LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32))))
        model.add(Dropout(hp.Float('dropout_2', 0.1, 0.5, step=0.1)))
        model.add(Dense(self.num_outputs))  # Output layer with one neuron per forecast horizon
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')), 
                      loss='mean_squared_error')
        return model

# Initialize the HyperModel
hypermodel = TrafficPredictionHyperModel(input_shape=(X_train.shape[1], X_train.shape[2]), num_outputs=len(horizons))

# Initialize the RandomSearch tuner
tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=10,
    directory='random_search',
    project_name='traffic_prediction'
)

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Perform hyperparameter tuning
tuner.search(X_train, Y_train, epochs=50, validation_data=(X_test, Y_test), callbacks=[early_stopping])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the optimal hyperparameters and train it
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test), verbose=1, callbacks=[early_stopping])

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'Model Training and Validation Loss (Look-back period: {look_back})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions and true values
for i in range(len(horizons)):
    train_predict[:, i] = scalers['congestion_level'].inverse_transform(train_predict[:, i].reshape(-1, 1)).flatten()
    Y_train[:, i] = scalers['congestion_level'].inverse_transform(Y_train[:, i].reshape(-1, 1)).flatten()
    test_predict[:, i] = scalers['congestion_level'].inverse_transform(test_predict[:, i].reshape(-1, 1)).flatten()
    Y_test[:, i] = scalers['congestion_level'].inverse_transform(Y_test[:, i].reshape(-1, 1)).flatten()

# Calculate performance metrics for each horizon
results = []
for i, horizon in enumerate(horizons):
    train_mae = mean_absolute_error(Y_train[:, i], train_predict[:, i])
    test_mae = mean_absolute_error(Y_test[:, i], test_predict[:, i])
    train_mse = mean_squared_error(Y_train[:, i], train_predict[:, i])
    test_mse = mean_squared_error(Y_test[:, i], test_predict[:, i])
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_accuracy = calculate_accuracy(Y_train[:, i], train_predict[:, i])
    test_accuracy = calculate_accuracy(Y_test[:, i], test_predict[:, i])

    results.append((horizon, train_mae, test_mae, train_mse, test_mse, train_rmse, test_rmse, train_accuracy, test_accuracy))
    print(f'Horizon: {horizon * 15} minutes')
    print(f'Train MAE: {train_mae:.2f}')
    print(f'Test MAE: {test_mae:.2f}')
    print(f'Train MSE: {train_mse:.2f}')
    print(f'Test MSE: {test_mse:.2f}')
    print(f'Train RMSE: {train_rmse:.2f}')
    print(f'Test RMSE: {test_rmse:.2f}')
    print(f'Train Accuracy: {train_accuracy:.2f}')
    print(f'Test Accuracy: {test_accuracy:.2f}')

    # Plot predictions for this horizon
    plt.figure(figsize=(14, 7))
    plt.plot(data.index[-len(test_predict):], Y_test[:, i], label='True Congestion Level')
    plt.plot(data.index[-len(test_predict):], test_predict[:, i], label='Predicted Congestion Level')
    plt.title(f'Predicted vs True Congestion Level for {horizon * 15} minutes (Look-back period: {look_back})')
    plt.xlabel('Timestamp')
    plt.ylabel('Congestion Level')
    plt.legend()
    plt.show()
    
# Correctly retrieve the learning rate from the HyperParameters object
learning_rate = best_hps.values['learning_rate']

# Then, compile the model with the corrected learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='mean_squared_error')


# Save the model using the recommended Keras format
model.save('traffic_model.keras')

#save the scalers
joblib.dump(scalers, 'scalers.pkl')
