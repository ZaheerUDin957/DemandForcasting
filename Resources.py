# Growth (Trend)
**What it is:** The underlying trend or direction of your data over time.  
**Options:**
- **Linear:** A steady increase or decrease over time.
- **Logistic:** Growth that follows an S-shaped curve, where the trend grows quickly at first and then levels off.
- **Flat:** No growth, the values remain constant over time.

# Changepoints
**What it is:** Specific points in time where the trend in your data might change (e.g., due to a policy change, event, or other significant factor).  
**Options:**
- **Manual changepoints:** You can manually specify dates where you expect changes in the trend.
- **Automatic selection:** Prophet automatically detects changepoints in the early part of your data based on patterns it identifies.

# Number of Changepoints (`n_changepoints`)
**What it is:** If you don't provide your own changepoints, Prophet will try to find changes in the trend.  
**Use case:** When the model automatically selects changepoints, you can set the number of changepoints to use. Prophet typically selects these changepoints within the first 80% of the data (by default).

# Changepoint Range
**What it is:** This defines the portion of your data where the model will search for changepoints.  
**Default:** 80% of the historical data is used for detecting changepoints, meaning changepoints will be identified in the first 80% of the data.

# Seasonality
**What it is:** Recurring patterns that happen over specific intervals like days, weeks, months, or years.  
**Options:**
- **Yearly Seasonality:** Repeats annually, capturing things like holiday effects or season changes.
- **Weekly Seasonality:** Captures patterns that repeat every week (e.g., different behavior on weekdays vs weekends).
- **Daily Seasonality:** Captures patterns that repeat every day, such as hourly trends.  
**Settings:** You can either let the model automatically detect the seasonality (`auto`), enforce it (`True`), or disable it (`False`). You can also specify how many Fourier terms to include for the seasonality components.

# Holidays
**What it is:** Significant days or events that might affect the forecast (e.g., holidays, national events).  
**How it works:** You can pass a DataFrame containing the holidays or events to include in the model. Prophet can consider how these holidays affect the trend and seasonality patterns.

# Seasonality Mode
**What it is:** The way the seasonal component interacts with the overall trend.  
**Options:**
- **Additive (default):** The seasonal effects are added to the trend. This is useful when the seasonality does not change dramatically with the level of the trend.
- **Multiplicative:** The seasonal effects scale with the level of the trend. This is useful when higher values have larger seasonal effects.

# Forecasting Parameters:
- **Initial period:** The initial amount of data used for training the model. It helps define how much of the data is used before the first forecast is made.
- **Horizon:** How far into the future you want to predict.
- **Period:** Defines how often new forecasts should be made within a time period.
- **Uncertainty interval:** The range of uncertainty around the predictions, which provides a confidence range.





  # Import necessary libraries
from fbprophet import Prophet
import pandas as pd

# Function to create and configure a Prophet model
def create_prophet_model(growth='linear', changepoint_range=0.8, yearly_seasonality='auto', 
                         weekly_seasonality='auto', daily_seasonality=False, 
                         seasonality_mode='additive', n_changepoints=25):
    """
    Create and configure a Prophet model with specified parameters.
    
    Parameters:
    - growth: 'linear' for linear trend, 'logistic' for logistic growth.
    - changepoint_range: Proportion of data used for detecting changepoints (default is 80%).
    - yearly_seasonality: Whether to include yearly seasonality ('auto', True, False).
    - weekly_seasonality: Whether to include weekly seasonality ('auto', True, False).
    - daily_seasonality: Whether to include daily seasonality (True or False).
    - seasonality_mode: 'additive' or 'multiplicative' seasonality.
    - n_changepoints: Number of potential changepoints to include.
    
    Returns:
    - A configured Prophet model.
    """
    model = Prophet(growth=growth,
                    changepoint_range=changepoint_range,
                    yearly_seasonality=yearly_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    daily_seasonality=daily_seasonality,
                    seasonality_mode=seasonality_mode,
                    n_changepoints=n_changepoints)
    
    return model

# Function to add custom holidays
def add_holidays_to_model(model, holidays_df):
    """
    Add holidays to a Prophet model.
    
    Parameters:
    - model: The Prophet model.
    - holidays_df: DataFrame with holiday information (columns: 'ds', 'holiday').
    
    Returns:
    - The Prophet model with holidays added.
    """
    model.add_country_holidays(country_name='US')  # Example: US holidays, modify as per requirement
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # Example: monthly seasonality
    
    return model

# Function to fit the model to data
def fit_prophet_model(model, df):
    """
    Fit a Prophet model to data.
    
    Parameters:
    - model: The Prophet model.
    - df: DataFrame containing 'ds' (date) and 'y' (value) columns.
    
    Returns:
    - The fitted Prophet model.
    """
    model.fit(df)
    return model

# Function to make future predictions
def make_future_predictions(model, periods, freq='D'):
    """
    Make future predictions using a fitted Prophet model.
    
    Parameters:
    - model: The fitted Prophet model.
    - periods: Number of periods to predict into the future.
    - freq: Frequency of the predictions (default is daily).
    
    Returns:
    - A DataFrame containing future predictions.
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

# Function to plot forecast results
def plot_forecast(model, forecast):
    """
    Plot forecast results using Prophet's built-in plotting functions.
    
    Parameters:
    - model: The Prophet model.
    - forecast: DataFrame containing forecasted data.
    
    Returns:
    - A plot of the forecast.
    """
    fig = model.plot(forecast)
    return fig

# Example function to run the entire forecasting process
def run_forecasting_pipeline(data, growth='linear', changepoint_range=0.8, yearly_seasonality='auto', 
                             weekly_seasonality='auto', daily_seasonality=False, seasonality_mode='additive',
                             n_changepoints=25, periods=365, freq='D', holidays_df=None):
    """
    Run the complete Prophet forecasting pipeline.
    
    Parameters:
    - data: DataFrame containing historical data with 'ds' (date) and 'y' (value).
    - growth: Trend type ('linear' or 'logistic').
    - changepoint_range: Portion of data to use for changepoint detection.
    - yearly_seasonality: Yearly seasonality setting.
    - weekly_seasonality: Weekly seasonality setting.
    - daily_seasonality: Daily seasonality setting.
    - seasonality_mode: Additive or multiplicative seasonality.
    - n_changepoints: Number of changepoints to use.
    - periods: Number of future periods to predict.
    - freq: Frequency of predictions (e.g., 'D' for daily).
    - holidays_df: Optional DataFrame with holiday data.
    
    Returns:
    - Forecasted results as a DataFrame.
    """
    # Step 1: Create and configure the model
    model = create_prophet_model(growth, changepoint_range, yearly_seasonality, 
                                 weekly_seasonality, daily_seasonality, 
                                 seasonality_mode, n_changepoints)
    
    # Step 2: Add holidays if provided
    if holidays_df is not None:
        model = add_holidays_to_model(model, holidays_df)
    
    # Step 3: Fit the model to historical data
    model = fit_prophet_model(model, data)
    
    # Step 4: Make future predictions
    forecast = make_future_predictions(model, periods, freq)
    
    # Step 5: Plot the forecast
    plot_forecast(model, forecast)
    
    return forecast

