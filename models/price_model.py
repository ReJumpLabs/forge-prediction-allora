import sqlite3
from contextlib import closing

VALID_TOKENS = ['ETH', 'SOL', 'BTC', 'BNB', 'ARB']

db_path = 'database/prices.db'

def save_price_to_db(symbol, price):
    symbol = str(symbol).upper()
    if symbol not in VALID_TOKENS:
        print(f"Token {symbol} không hợp lệ.")
        return

    conn = sqlite3.connect(db_path)
    with closing(conn.cursor()) as cursor:
        cursor.execute("SELECT 1 FROM prices WHERE symbol = ?", (symbol,))
        exists = cursor.fetchone()

        if exists:
            cursor.execute("UPDATE prices SET price = ?, timestamp = CURRENT_TIMESTAMP WHERE symbol = ?", (price, symbol))
            print(f"Đã cập nhật giá của {symbol} thành {price}")
        else:
            print(f"Đã thêm {symbol} với giá {price}")
            cursor.execute("INSERT INTO prices (symbol, price) VALUES (?, ?)", (symbol, price))
        
    conn.commit()
    conn.close()

def get_latest_price(symbol):
    conn = sqlite3.connect('database/prices.db')
    with closing(conn.cursor()) as cursor:
        cursor.execute("SELECT price FROM prices WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1", (symbol,))
        result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def predict_usdc_eth_volume():
    """
    Predict the trading volume of USDC/ETH.
    """
    # Load or update the model
    interpreter = load_or_update_model('ETH')

    # Load and preprocess data
    data = load_data('ETH')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    # Prepare input data
    last_60_days = scaled_data[-60:]
    x_input = np.array([last_60_days])
    x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))

    # Predict future volumes using TFLite interpreter
    predictions = []
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for _ in range(5):  # Predict volume for the next 5 minutes
        interpreter.set_tensor(input_details[0]['index'], x_input.astype(np.float32))
        interpreter.invoke()
        predicted_volume = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(predicted_volume[0, 0])
        # Update x_input with new prediction
        predicted_volume_reshaped = np.array(predicted_volume).reshape((1, 1, 1))
        x_input = np.append(x_input[:, 1:, :], predicted_volume_reshaped, axis=1)

    # Scale predictions back to original values
    final_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    return final_predictions[-1][0]
