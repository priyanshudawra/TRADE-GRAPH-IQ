import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
import matplotlib.pyplot as plt
import io
import base64

# Set the matplotlib backend to 'Agg' (non-GUI) to avoid threading issues on macOS
matplotlib.use('Agg')

def train_and_evaluate_model(stock_name):
    # Download historical data
    stock_data = yf.download(stock_name, period='1y').dropna()

    # Normalize and prepare features
    scaler = MinMaxScaler()
    stock_data_normalized = scaler.fit_transform(stock_data[['Close']])
    stock_features = stock_data[['Open', 'High', 'Low', 'Volume']].assign(Close=stock_data_normalized)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        stock_features, stock_data['Close'], test_size=0.2, random_state=42
    )

    # Train the SVM model
    svm_model = SVR(kernel='rbf', C=100, gamma=0.1)
    svm_model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = svm_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Generate the graph for the stock's closing prices
    graph_url = plot_stock_data(stock_data, stock_name)

    return mse, rmse, r2, svm_model, scaler, graph_url

def predict_new_data(stock_name, svm_model, scaler, limit=1):
    # Load recent stock data with a valid period (e.g., '5d')
    new_stock_data = yf.download(stock_name, period='5d').dropna()

    # Handle empty data case
    if new_stock_data.empty:
        return []

    # Normalize and prepare features for prediction
    new_stock_data_normalized = scaler.transform(new_stock_data[['Close']])
    new_stock_features = new_stock_data[['Open', 'High', 'Low', 'Volume']].assign(Close=new_stock_data_normalized)

    # Predict on new data
    new_prediction = svm_model.predict(new_stock_features)
    return [new_prediction[-1]]  # Single prediction for the most recent data

def plot_stock_data(stock_data, stock_name):
    # Ensure the plotting happens in the main thread
    plt.figure(figsize=(5, 4))  # Set a smaller size to fit within the white theme box
    plt.plot(stock_data['Close'], label=f'{stock_name} Closing Prices')
    plt.title(f'{stock_name} Stock Price Over Last Year')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Adjust layout to prevent clipping of labels and titles
    plt.tight_layout()

    # Convert plot to PNG image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Encode image to base64 to send to HTML
    graph_url = base64.b64encode(image_png).decode('utf-8')
    plt.close()  # Close the plot to free up memory

    return graph_url
