import tkinter as tk
import tensorflow as tf
import pandas as pd
import quandl

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, 4)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Function to load the training data for a given stock
def load_data(stock):
    data = quandl.get('WIKI/{}'.format(stock), start_date='2000-01-01', end_date='2010-12-31')
    x_train = data[['Open', 'High', 'Low', 'Close']].rolling(100).mean().dropna().values.reshape(-1, 100, 4)
    y_train = (data['Close'].shift(-1) > data['Close']).astype(int).values[100:]
    return x_train, y_train

# Function to train the model on the selected stock
def train_model(stock):
    x_train, y_train = load_data(stock)
    model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Create the main window
window = tk.Tk()
window.title('Trading Bot Training')

# Add a label and text field for entering the stock symbol
stock_label = tk.Label(window, text='Stock Symbol:')
stock_label.grid(row=0, column=0)
stock_entry = tk.Entry(window)
# Configure the text field to automatically select the text when focused
stock_entry.bind('<FocusIn>', lambda event: stock_entry.selection_range(0, 'end'))

# Add a button for training the model on the selected stock
train_button = tk.Button(window, text='Train Model', command=lambda: train_model(stock_entry.get()))
train_button.grid(row=1, column=0, columnspan=2)

# Add a label and progress bar for showing the training progress
progress_label = tk.Label(window, text='Training Progress:')
progress_label.grid(row=2, column=0)
progress_bar = tk.ttk.Progressbar(window, orient='horizontal', length=200, mode='determinate')
progress_bar.grid(row=2, column=1)

# Function to update the progress bar during training
def update_progress(epoch, logs):
    progress_bar['value'] = (epoch+1)/100*100

# Attach the progress update function to the on_epoch_end event
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=update_progress)])

# Add a button for saving the trained model
save_button = tk.Button(window, text='Save Model', command=lambda: model.save('my_trading_bot.h5'))
save_button.grid(row=3, column=0, columnspan=2)

# Run the main event loop
window.mainloop()

