import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Assuming '/content/drive/MyDrive' is a valid path, modify if necessary
file_path = '/content/drive/MyDrive/CCD.xls'
df = pd.read_excel(file_path)

# Prepare data by grouping
array1 = df[['X12', 'X13', 'X14', 'X15', 'X16', 'X17']].apply(pd.to_numeric, errors='coerce').sum(axis=1)
array2 = df[['X18', 'X19', 'X20', 'X21', 'X22', 'X23']].apply(pd.to_numeric, errors='coerce').sum(axis=1)

df.drop(['X12', 'X13', 'X14', 'X15', 'X16', 'X17'], axis=1, inplace=True)
df.drop(['X18', 'X19', 'X20', 'X21', 'X22', 'X23'], axis=1, inplace=True)

df.insert(11, 'X12', array1)
df.insert(12, 'X13', array2)

# Preprocess the data
x = df.drop(columns=['Y']).iloc[1:].apply(pd.to_numeric, errors='coerce')
y = df['Y'].iloc[1:].astype('float32')

# One hot encode colomn as it may think 2 is better than 1 but won't for 1 and 0
x['X2'] = x['X2'].map({2: 1, 1: 0}).astype('float32')

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler().fit(x_train)  # Fit on training data only
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Build the model
model = Sequential()
model.add(Dense(units=256, activation='tanh', input_dim=x_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='SGD', loss='hinge', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=25, batch_size=256, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy*100:.2f}%')

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
