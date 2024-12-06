import argparse
import os
import asyncio
from quart import Quart, jsonify, Response
from models.lstm_model import predict_price, fetch_binance_data, save_to_csv, train_model
app = Quart(__name__)

semaphore = asyncio.Semaphore(200)

@app.route("/collect-price")
async def train_model_endpoint():
    """
    Endpoint to train the model for all supported tokens.
    
    :return: Response, JSON response indicating the status
    """
    try:
        supported_tokens = ['ETH', 'BTC', 'SOL', 'BNB', 'ARB']
        
        for token in supported_tokens:
            token = token.upper()
            
            # Ensure data exists
            if not os.path.exists(f'data/{token}_data.csv'):
                print(f"Fetching data for {token}...")
                data = await fetch_binance_data(token)
                save_to_csv(data, token)
            
            # Train the model
            print(f"Training model for {token}...")
            train_model(token)
        
        return Response("Models for all supported tokens trained successfully", status=200, mimetype='application/json')
    
    except Exception as e:
        print(f"Error while training models: {str(e)}")
        return Response(f"Failed to train models. Error: {str(e)}", 
                        status=500, mimetype='application/json')

@app.route("/inference/<string:token>")
async def get_inference(token):
    try:
        await semaphore.acquire()

        predict_result = None
        token = token.upper()

        if token:

            predict_result = await predict_price(token)

        else:
            return jsonify({"error": "Token is not supported"}), 500

        if predict_result is None:
            return jsonify({"error": "Failed to generate prediction"}), 500

        return predict_result

    finally:
        semaphore.release()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=5000, type=int, help="Port to run the app on")
    args = parser.parse_args()
    
    app.run(host="0.0.0.0", port=args.port)
