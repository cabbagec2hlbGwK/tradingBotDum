import yfinance as yf
import pandas as pd
import tensorflow as tf

# Get the stock data for Tesla
tsla = yf.Ticker("TSLA")

# Get the historical data for the stock
historical_data = tsla.history(period="max")

# Select the relevant columns
X = historical_data[['Open', 'High', 'Low', 'Close', 'Volume']]
y = historical_data['Close'].shift(-1)  # The target is the next day's closing price

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Preprocess the data (e.g., handle missing values, normalize numerical features)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)

# Make predictions on new data
new_data = np.load('new_data.npy')
predictions = model.predict(new_data)
