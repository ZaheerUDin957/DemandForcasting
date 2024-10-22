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
