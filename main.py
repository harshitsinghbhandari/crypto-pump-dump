import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import time

# Import project modules
from data.data_collection import fetch_binance_historical_data
from features.feature_engineering import extract_features
from models.model_development import train_isolation_forest
from utils.utils import setup_logging, plot_anomalies
from reports.report_generation import generate_report

def main():
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Crypto Pump & Dump Detection System")
    
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    try:
        # Fetch historical data for a cryptocurrency
        logger.info("Fetching historical data from Binance")
        symbol = 'BTCUSDT'  # Bitcoin/USDT pair
        interval = '1h'     # 1-hour intervals
        
        # Calculate timestamps (last 30 days)
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
        
        # Fetch data
        data = fetch_binance_historical_data(symbol, interval, start_time, end_time)
        
        # Save raw data
        raw_data_path = f'data/raw/{symbol}_{interval}_{datetime.now().strftime("%Y%m%d")}.csv'
        data.to_csv(raw_data_path, index=False)
        logger.info(f"Raw data saved to {raw_data_path}")
        
        # Extract features
        logger.info("Extracting features")
        features_df = extract_features(data)
        
        # Save processed data
        processed_data_path = f'data/processed/{symbol}_{interval}_processed_{datetime.now().strftime("%Y%m%d")}.csv'
        features_df.to_csv(processed_data_path, index=False)
        logger.info(f"Processed data saved to {processed_data_path}")
        
        # Train anomaly detection model
        logger.info("Training anomaly detection model")
        feature_columns = ['close_z_score', 'volume_z_score', 'relative_volume', 
                          'price_velocity', 'price_acceleration', 'volatility']
        
        model, results = train_isolation_forest(features_df, feature_columns, contamination=0.03)
        
        # Save results
        results_path = f'data/processed/{symbol}_{interval}_results_{datetime.now().strftime("%Y%m%d")}.csv'
        results.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        # Plot anomalies
        logger.info("Generating anomaly plot")
        plot_anomalies(results)
        
        # Generate report
        logger.info("Generating project report")
        generate_report()
        
        logger.info("Crypto Pump & Dump Detection System completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
