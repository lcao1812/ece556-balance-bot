import matplotlib.pyplot as plt
from keras.api.saving import register_keras_serializable
import joblib
from keras.api.callbacks import EarlyStopping
from keras.api.layers import Dense, Conv1D, Flatten, Dropout
from keras import Sequential
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import os

# Adjust the Python path to include the spkeras module
currentpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(currentpath, '../spkeras'))


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
    # Penalize large acceleration values
    forward_penalty = tf.reduce_mean(tf.square(y_pred))
    cumulative_displacement_penalty = tf.reduce_mean(
        tf.square(X_train[:, 4]))  # Penalize x_position displacement
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    # Combine MSE with penalties
    return mse_loss + 0.1 * forward_penalty + 0.3 * cumulative_displacement_penalty


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

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {loss}, Test MAE: {mae}')

# Save the model in the new format
model.save('cnn.keras')
print('Model saved as cnn.keras')

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.axhline(y=loss, color='r', linestyle='--', label='Test Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
# Save the plot to the specified directory
output_dir = os.path.join(currentpath, '../Report')
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, 'training_validation_loss.png')
plt.savefig(plot_path)
print(f'Plot saved at {plot_path}')