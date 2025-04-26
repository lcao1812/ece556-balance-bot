import sys
import os

# Adjust the Python path to include the spkeras module
currentpath = os.path.dirname(os.path.realpath(__file__))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from tensorflow.keras.saving import register_keras_serializable

# Load the dataset
data = pd.read_csv('training_data.csv')

# Extract features and target
X = data[['theta', 'omega', 'velocity', 'targetvelocity', 'x_position']].values
y = data['acc'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Ensure the scaler is trained on all 5 features, including x_position
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print('Scaler saved as scaler.pkl')

# Reshape the data for Conv1D input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Ensure all input data is converted to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')


# Register the custom loss function
@register_keras_serializable()
def custom_loss(y_true, y_pred):
    forward_penalty = tf.reduce_mean(tf.square(y_pred))  # Penalize large acceleration values
    cumulative_displacement_penalty = tf.reduce_mean(tf.square(X_train[:, 4]))  # Penalize x_position displacement
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    return mse_loss + 0.1 * forward_penalty + 0.3 * cumulative_displacement_penalty  # Combine MSE with penalties


# Build the CNN model
model = Sequential([
    Conv1D(32, kernel_size=2, activation='relu',
           input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    Conv1D(64, kernel_size=2, activation='relu'),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss=custom_loss, metrics=['mae'])

# Train the model with early stopping
early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_split=0.4,
    epochs=500,  # Increased from 200 to 500
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# I could not load the model with the custom loss function
# so I recreated the model here and use the testing data instead of the validation data

# Load testing data from CSV
# Ensure the CSV file has the same structure as the training data
# Example: ['theta', 'omega', 'velocity', 'targetvelocity', 'acc', 'x_position']
testing_data_df = pd.read_csv('testing_data.csv')

# Extract features and labels
testing_data = testing_data_df[['theta', 'omega', 'velocity', 'targetvelocity', 'x_position']].values
labels = testing_data_df['acc'].values

# Preprocess the testing data
# Ensure the scaler used during training is applied here
testing_data = scaler.transform(testing_data)

# Reshape the data for the model
testing_data = testing_data.reshape((testing_data.shape[0], testing_data.shape[1], 1))

# Evaluate the model on the testing data
loss, mae = model.evaluate(testing_data, labels, verbose=2)

print(f"Test Loss: {loss}")
print(f"Test MAE: {mae}")

