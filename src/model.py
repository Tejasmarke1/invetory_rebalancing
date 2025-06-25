import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import datetime

def train_model(features_df):
    predictions = []

    # Get the latest week to forecast for the next one
    last_week = features_df['Week'].max()
    next_week = last_week + pd.Timedelta(weeks=1)

    # Loop over each SKU-Zone pair
    for (sku, zone), group in features_df.groupby(['SKU', 'Zone']):
        # Prepare data
        group = group.sort_values('Week')
        X = group[['Trend_Score', 'Is_Holiday']]
        y = group['Quantity_Sold']

        # Train-test split (optional in production)
        if len(X) < 4:  # Not enough data
            continue
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Model training
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict next week's trend and holiday (use last known values)
        last_trend = X.iloc[-1]['Trend_Score']
        last_holiday = X.iloc[-1]['Is_Holiday']

        X_future = pd.DataFrame([[last_trend * 1.05, last_holiday]], columns=['Trend_Score', 'Is_Holiday'])
        forecast = model.predict(X_future)[0]

        predictions.append({
            'SKU': sku,
            'Zone': zone,
            'Predicted_Week': next_week.date(),
            'Forecast_Quantity': round(forecast)
        })

    return pd.DataFrame(predictions)
