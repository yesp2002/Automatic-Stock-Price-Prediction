import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import date, timedelta,  datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# Streamlit app
def main():
    # st.set_page_config(layout="wide")
    # background_style = """
    #                 <style>
    #                 .stApp {
    #                     /* Add your background image URL */
    #                     background-color: #f0f0f0; /* Add your background color */
    #                     background-size: cover;
    #                     background-repeat: no-repeat;
    #                 }
    #                 </style>
    #             """

    # st.markdown(background_style, unsafe_allow_html=True)
    
    st.title("Stock Prediction Dashboard")
    st.markdown("---")
    # Define the company names
    company_names = ['Amazon', 'Apple', 'Microsoft', 'Google', 'Facebook']

    # Create a dropdown menu for stock selection
    selected_company = st.sidebar.selectbox("Select a company", company_names)

    # Map the company names to their corresponding stock symbols
    company_symbols = {
        'Amazon': 'AMZN',
        'Apple': 'AAPL',
        'Microsoft': 'MSFT',
        'Google': 'GOOGL',
        'Facebook': 'FB'
    }

    # Get the stock symbol for the selected company
    selected_symbol = company_symbols.get(selected_company)

    # Get the stock data for the selected company
    if selected_symbol:
        ticker = yf.Ticker(selected_symbol)
        start_date = datetime.now() - timedelta(days=2 * 365)
        end_date = datetime.now()
        stock_data = ticker.history(start=start_date, end=end_date)

        if not stock_data.empty:
            # Display the stock data graph
            # col1, col2 = st.columns(2)
            # with col1:
            st.subheader("Stock Data")
            # st.line_chart(stock_data[['Close', 'Open']])
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Open'], mode='lines', name='Open Price'))
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))

            # Customize the gridline color
            fig.update_layout(yaxis=dict(gridcolor='lightgray'), xaxis=dict(gridcolor='lightgray'))

            fig.update_layout(title= f"{selected_company} Stock Price (Open and Close) Line Plot",
                            xaxis_title="Date",
                            yaxis_title="Stock Price",
                            title_x=0.1,  # Title centered in the middle
                            title_yanchor='middle')

            st.plotly_chart(fig, use_container_width=True)
        
        # with col2:
            ticker_symbol = selected_symbol

            # Retrieve historical stock data using yfinance
            end_date = date.today() - timedelta(days=1)  # Exclude current day
            start_date = end_date - timedelta(days=2*365)  # Retrieve data for the past year
            data = yf.download(ticker_symbol, start=start_date, end=end_date)

            data = data[['Close']]
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            X_train = []
            y_train = []

            # Use the previous one year's data to predict a day's stock for each LSTM layer
            lookback = 3

            for i in range(lookback, len(scaled_data)):
                X_train.append(scaled_data[i - lookback: i, 0])
                y_train.append(scaled_data[i, 0])

            X_train, y_train = np.array(X_train), np.array(y_train)

            # Reshape the training data for LSTM input
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(units=365, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(LSTM(units=100, return_sequences=True))
            model.add(LSTM(units=52, return_sequences=False))
            model.add(Dense(units=1))

            # Compile and train the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=10, batch_size=32)

            # Get today's stock price
            today = date.today()
            today_data = yf.download(ticker_symbol, start=today, end=today)
            today_stock_price = today_data['Close'].iloc[0]

            # Prepare the input data for prediction
            last_year_data = data.tail(lookback)
            inputs = scaler.transform(last_year_data)

            X_test = []
            X_test.append(inputs)
            X_test = np.array(X_test)

            # Reshape the test data for LSTM input
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            # Predict today's stock price
            predicted_stock_price = model.predict(X_test)
            predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

            # Compare predicted price with today's stock price
            print(f"Predicted stock price for today: {predicted_stock_price[0][0]}")
            print(f"Actual stock price for today: {today_stock_price}")

            # Calculate the difference between predicted and actual prices
            price_difference = predicted_stock_price[0][0] - today_stock_price
            if price_difference > 0:
                print(f"The predicted price is higher than the actual price by {price_difference}")
            elif price_difference < 0:
                print(f"The predicted price is lower than the actual price by {-price_difference}")
            else:
                print("The predicted price is equal to the actual price")

            st.subheader("Prediction vs Actual")
            st.write("Today's Predicted Value: ", predicted_stock_price[0][0])
            st.write("Today's Actual Value: ", today_stock_price)

            st.markdown("---")
            # Calculate insights based on stock graph
            st.subheader("Insights")
            if price_difference > 0:
                st.write("The stock price is predicted to increase today.")
            else:
                st.write("The stock price is predicted to decrease today.")
                        

if __name__ == '__main__':
    main()
