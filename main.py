import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from flask import Flask, request, render_template_string
import plotly.graph_objects as go
import traceback # Added for more detailed error logging

app = Flask(__name__)

# Define HTML template directly in the script
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Discharge Flags</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h2>Discharge Data for Site ID: {{ site_id }}</h2>
    <div id="plot"></div>

    <script>
        // Get plot data passed from Flask template
        var plot_data = {{ plot_json | tojson | safe }};
        // Create the plot in the 'plot' div
        Plotly.newPlot('plot', plot_data.data, plot_data.layout);
    </script>
</body>
</html>
"""

@app.route('/plot')
def plot_site():
    # Get site ID from query parameter ?id=...
    site_id = request.args.get("id")
    if not site_id:
        return "Missing 'id' parameter in query string. Please provide a site ID like /plot?id=YOUR_SITE_ID", 400

    # --- Data Fetching ---
    try:
        # Get current date for API request
        end_date = datetime.today().strftime("%Y-%m-%d")
        # Construct API URL
        # Note: Current date is April 7, 2025 (as per context)
        api_url = f"https://www.waterrights.utah.gov/dvrtdb/daily-chart.asp?station_id={site_id}&end_date={end_date}&f=json"

        response = requests.get(api_url, timeout=20) # Increased timeout slightly
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # Validate structure of received data
        if "data" not in data:
            return f"API response for site {site_id} is missing the 'data' key. Response: {response.text[:500]}...", 500
        if not isinstance(data["data"], list):
             return f"API response 'data' field is not a list for site {site_id}. Response: {response.text[:500]}...", 500
        # Allow empty data list, handle later in processing
        # if not data["data"]:
        #      return f"API response for site {site_id} contains an empty data list.", 500

    except requests.exceptions.Timeout:
         return f"API request timed out for site {site_id} at {api_url}", 504 # Gateway Timeout
    except requests.exceptions.HTTPError as http_err:
         return f"HTTP error occurred accessing API for site {site_id}: {http_err}. Response: {response.text[:500]}...", 502 # Bad Gateway or appropriate error
    except requests.exceptions.RequestException as req_err:
        return f"API request error for site {site_id}: {str(req_err)}", 500
    except ValueError: # Includes JSONDecodeError
        return f"Failed to decode JSON from API response for site {site_id}. Content: {response.text[:500]}...", 500
    except Exception as e: # Catch unexpected errors during fetch
        tb_str = traceback.format_exc()
        return f"An unexpected error occurred during data fetching for site {site_id}: {str(e)}\n{tb_str}", 500

    # --- Data Processing ---
    try:
        # Create DataFrame if data list is not empty
        if not data["data"]:
             df = pd.DataFrame(columns=["Date", "DISCHARGE"]) # Create empty df with expected columns
        else:
            df = pd.DataFrame(data["data"], columns=["date", "value"])
            df.rename(columns={"date": "Date", "value": "DISCHARGE"}, inplace=True)

        # Convert types and handle errors
        df["DISCHARGE"] = pd.to_numeric(df["DISCHARGE"], errors='coerce')
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

        # Drop rows where essential data is missing AFTER conversion attempts
        df = df.dropna(subset=['DISCHARGE', 'Date'])

        # Check if DataFrame is empty after cleaning
        if df.empty:
            return f"No valid discharge data points found for site {site_id} after cleaning API response.", 200 # Return OK but indicate no data to plot

        # Extract Metadata
        metadata_fields = ["station_id", "station_name", "system_name", "units"]
        metadata = {field: data.get(field, "N/A") for field in metadata_fields}

        # --- Flagging Logic ---
        df['FLAG_NEGATIVE'] = (df['DISCHARGE'] < 0) & (df['DISCHARGE'] != 0)
        df['FLAG_ZERO'] = (df['DISCHARGE'] == 0)

        # Calculate flags that depend on non-zero data distribution
        non_zero_discharge = df.loc[df['DISCHARGE'] != 0, 'DISCHARGE']

        if not non_zero_discharge.empty:
            # Percentile-based flags
            discharge_95 = np.percentile(non_zero_discharge, 95)
            df['FLAG_Discharge'] = (df['DISCHARGE'] > discharge_95) & (df['DISCHARGE'] != 0)

            # IQR-based flags
            Q1, Q3 = non_zero_discharge.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            # Avoid IQR=0 issues if Q1==Q3
            if IQR > 0:
                 lower_bound = Q1 - 1.5 * IQR
                 upper_bound = Q3 + 1.5 * IQR
                 df['FLAG_IQR'] = ((df['DISCHARGE'] < lower_bound) | (df['DISCHARGE'] > upper_bound)) & (df['DISCHARGE'] != 0)
            else:
                 # If IQR is 0, only flag values different from Q1 (which equals Q3)
                 df['FLAG_IQR'] = (df['DISCHARGE'] != Q1) & (df['DISCHARGE'] != 0)


            # Rate of Change flags
            df['RATE_OF_CHANGE'] = df['DISCHARGE'].diff().abs()
            df['FLAG_RoC'] = (df['RATE_OF_CHANGE'] > discharge_95) & (df['DISCHARGE'] != 0) & (~df['RATE_OF_CHANGE'].isna())

            # Isolation Forest Outliers (only on non-zero values)
            df_clean = df.loc[df['DISCHARGE'] != 0, ['DISCHARGE']].copy() # Select only the column needed
            if not df_clean.empty and len(df_clean) > 1: # IF needs at least 2 points
                 model = IsolationForest(contamination='auto', random_state=42) # Use 'auto' contamination
                 df_clean['OUTLIER_IF_pred'] = model.fit_predict(df_clean[['DISCHARGE']])
                 # Map predictions back to original DataFrame
                 df['OUTLIER_IF'] = False # Initialize column
                 df.loc[df_clean.index, 'OUTLIER_IF'] = (df_clean['OUTLIER_IF_pred'] == -1)
            else:
                df['OUTLIER_IF'] = False # Not enough non-zero data for IF
        else:
            # If all discharge values are zero or NaN, set these flags to False
            df['FLAG_Discharge'] = False
            df['FLAG_IQR'] = False
            df['FLAG_RoC'] = False
            df['OUTLIER_IF'] = False
            df['RATE_OF_CHANGE'] = np.nan # RoC cannot be calculated


        # Repeated Value flags (only check non-zero repeats)
        # Identify groups of consecutive identical values, calculate size, flag if >= 3 and non-zero
        non_zero_groups = df.loc[df['DISCHARGE'] != 0, 'DISCHARGE'].groupby((df['DISCHARGE'] != df['DISCHARGE'].shift()).cumsum())
        group_sizes = non_zero_groups.transform('size')
        df['FLAG_REPEATED'] = (group_sizes >= 3) & (df['DISCHARGE'] != 0)
        # Fill potential NaNs introduced (e.g., at the start or where discharge is zero)
        df['FLAG_REPEATED'] = df['FLAG_REPEATED'].fillna(False)


        # Consolidate: Determine if any flag is active for each point
        flag_cols_for_any = [
             'FLAG_NEGATIVE', 'FLAG_ZERO', 'FLAG_REPEATED',
             'FLAG_IQR', 'OUTLIER_IF', 'FLAG_Discharge', 'FLAG_RoC'
        ]
        # Ensure all expected flag columns exist before combining them
        for col in flag_cols_for_any:
            if col not in df.columns:
                df[col] = False # Initialize if calculation was skipped
        df['FLAGGED'] = df[flag_cols_for_any].any(axis=1)

    except Exception as e:
         # Catch errors during pandas/sklearn processing
         tb_str = traceback.format_exc()
         return f"An error occurred during data processing for site {site_id}: {str(e)}\nTraceback:\n{tb_str}", 500

    # --- Plotting ---
    try:
        fig = go.Figure()

        # Convert Date to string format suitable for Plotly JSON serialization just before plotting
        df["Date_str"] = df["Date"].dt.strftime('%Y-%m-%dT%H:%M:%S') # ISO format is robust

        # Add main discharge line trace
        fig.add_trace(go.Scatter(
            x=df['Date_str'].tolist(),
            y=df['DISCHARGE'].tolist(),
            mode='lines',
            line=dict(color='lightgray', width=1.5),
            name='Mean Daily Discharge' # Original name
        ))

        # Define flag colors and base names (used for legend)
        flag_colors = {
            'FLAG_NEGATIVE': ('red', 'Negative (-)'),
            'FLAG_ZERO': ('blue', 'Value = 0'),
            'FLAG_REPEATED': ('green', 'Repeated (≥3 days)'),
            'FLAG_RoC': ('brown', 'Rate of Change'),
            'FLAG_IQR': ('orange', 'IQR Outlier'),
            'OUTLIER_IF': ('teal', 'Isolation Forest'),
            'FLAG_Discharge': ('purple', 'Above 95th Percentile')
        }

        # Add traces for each type of flag marker
        for flag, (color, name) in flag_colors.items():
             # Check the flag column exists in the DataFrame
             if flag in df.columns:
                subset = df[df[flag]] # Select rows where this flag is True
                if not subset.empty:
                    fig.add_trace(go.Scatter(
                        x=subset['Date_str'].tolist(),
                        y=subset['DISCHARGE'].tolist(),
                        mode='markers',
                        marker=dict(color=color, size=7),
                        name=name # Original name from flag_colors dict
                    ))

        # Update layout: Title, Axis Labels, and Legend Position
        fig.update_layout(
            title=dict(
                text=f"Flagged Discharge Points - {metadata.get('station_name', f'Station {site_id}')}",
                x=0.5 # Center title
            ),
            yaxis_title=f"Mean Daily Discharge ({metadata.get('units', 'Unknown Units')})", # Use units from metadata
            template="plotly_white", # Cleaner plot background
            width=1400, # Plot width in pixels
            height=750, # Plot height - increased slightly for legend space
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom", # Anchor legend based on its bottom edge
                y=1.01,           # Position bottom edge slightly above plot area (y=1.0) - adjust as needed
                xanchor="center", # Anchor legend based on its horizontal center
                x=0.5             # Position center at the horizontal middle of plot area
            )
        )

        # Convert figure to dictionary for JSON serialization
        plot_json = fig.to_dict()

    except Exception as e:
        # Catch errors during Plotly figure generation/conversion
        tb_str = traceback.format_exc()
        return f"An error occurred during plot generation for site {site_id}: {str(e)}\n{tb_str}", 500

    # --- Rendering ---
    # Render the HTML template, passing plot data and site ID
    return render_template_string(HTML_TEMPLATE, plot_json=plot_json, site_id=site_id)


# --- Main execution block ---
if __name__ == '__main__':
    # This block is for local development ONLY.
    # For production deployment (like on GCP), use a WSGI server like Gunicorn or Waitress.
    # Example Gunicorn command: gunicorn --bind 0.0.0.0:$PORT main:app
    # (Assuming this file is named main.py and $PORT is set by the environment)

    # Run Flask's development server:
    print("Starting Flask development server...")
    print("Access the plot via http://localhost:8080/plot?id=YOUR_SITE_ID")
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
