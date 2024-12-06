import asyncio
import random
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from binance.client import Client
from tensorflow import lite

# Binance API keys (nên giữ trong biến môi trường)
BINANCE_API_KEY = ''
BINANCE_API_SECRET = ''

# Global variables to store models
MODELS = {}
MODEL_PATHS = {
    "ETH": "models/ETH_lstm_model.h5",
    "BTC": "models/BTC_lstm_model.h5",
    "SOL": "models/SOL_lstm_model.h5",
    "BNB": "models/BNB_lstm_model.h5",
    "ARB": "models/ARB_lstm_model.h5",
}


# ======= Helper Functions =======

async def fetch_binance_data(token):
    """
    Fetch Binance Kline data.
    """
    client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
    klines = client.get_klines(symbol=f'{token}USDT', interval=Client.KLINE_INTERVAL_1MINUTE, limit=1000)
    return klines


def save_to_csv(data, token):
    """
    Save data to CSV file.
    """
    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                     'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                     'taker_buy_quote_asset_volume', 'ignore'])
    df.to_csv(f'data/{token}_data.csv', index=False)


def load_data(token):
    """
    Load close prices from CSV file.
    """
    df = pd.read_csv(f'data/{token}_data.csv')
    df['close'] = df['close'].astype(float)
    return df['close'].values


def create_lstm_model(input_shape):
    """
    Create an LSTM model.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def convert_to_tflite(model, token):
    """
    Convert the Keras model to TensorFlow Lite format.
    """
    converter = lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_enable_resource_variables = True
    converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS, lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    with open(f'models/{token}_lstm_model.tflite', 'wb') as f:
        f.write(tflite_model)

def load_tflite_model(token):
    """
    Load the TensorFlow Lite model.
    """
    model_path = f'models/{token}_lstm_model.tflite'
    interpreter = lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def train_model(token):
    """
    Train and save LSTM model for a given token.
    """
    data = load_data(token)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    train_data = scaled_data[:int(len(scaled_data) * 0.8)]
    x_train, y_train = [], []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = create_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1)

    os.makedirs('models', exist_ok=True)
    model.save(f'models/{token}_lstm_model.h5')
    convert_to_tflite(model, token)


def load_or_update_model(token):
    """
    Load model if not already loaded, or update if necessary.
    """
    model_path = MODEL_PATHS.get(token)
    if not model_path:
        raise ValueError(f"No model path configured for token: {token}")

    if token not in MODELS:
        MODELS[token] = load_tflite_model(token)
        print(f"TFLite model for {token} loaded.")
    return MODELS[token]


# ======= Prediction Function =======

async def predict_price(token):
    """
    Predict the price of a token.
    """
    token = token.upper()
    prediction_horizon = 5  # Update to predict price after 5 minutes

    # Load or update the model
    interpreter = load_or_update_model(token)

    # Load and preprocess data
    data = load_data(token)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    # Prepare input data
    last_60_days = scaled_data[-60:]
    x_input = np.array([last_60_days])
    x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))

    # Predict future prices using TFLite interpreter
    predictions = []
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for _ in range(prediction_horizon):
        interpreter.set_tensor(input_details[0]['index'], x_input.astype(np.float32))
        interpreter.invoke()
        predicted_price = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(predicted_price[0, 0])
        # Update x_input with new prediction
        predicted_price_reshaped = np.array(predicted_price).reshape((1, 1, 1))
        x_input = np.append(x_input[:, 1:, :], predicted_price_reshaped, axis=1)

    # Scale predictions back to original values
    final_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Apply random fluctuation
    final_prediction = final_predictions[-1][0]
    fluctuation = final_prediction * random.uniform(-0.00015, 0.00015)
    final_prediction += fluctuation

    return final_prediction


