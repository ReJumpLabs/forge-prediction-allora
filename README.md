# Allora Prediction V5

## Setup

1. **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd Allora-Prediction-V5
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment:**
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Running the Application

1. **Run the application:**
    ```sh
    python app.py --port 5000
    ```

2. **Access the application:**
    Open your web browser and go to `http://localhost:5000`.

## Endpoints

- **Collect and save prices:**
    ```
    GET /collect-price
    ```

- **Get simple price:**
    ```
    GET /price/<token>
    ```

- **Get inference:**
    ```
    GET /inference/<token>
    ```

## Notes

- Ensure you have the necessary permissions to create and write to the `data` and `models` directories.
- The application uses Binance API to fetch data and CoinGecko API keys for price information.

