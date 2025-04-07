import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from flask import Flask, request, render_template_string
import plotly.graph_objects as go

app = Flask(__name__)

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
        var plot_data = {{ plot_json | tojson | safe }};
        Plotly.newPlot('plot', plot_data.data, plot_data.layout);
    </script>
</body>
</html>
"""

@app.route('/plot')
def plot_site():
    site_id = request.args.get("id")
    if not site_id:
        return "Missing 'id' parameter in query string", 400

    end_date = datetime.today().strftime("%Y-%m-%d")
    api_url = f"https://www.waterrights.utah.gov/dvrtdb/daily-chart.asp?station_id={site_id}&end_date={end_date}&f=json"

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "data" not in data:
            return f"No 'data' key in API response for site {site_id}", 500
        # Handle case where data list might be empty
        if not data["data"]:
             return f"API response for site {site_id} contains an empty data list", 500
    except requests.exceptions.RequestException as e:
        return f"API request error: {str(e)}", 500
    except ValueError:
        # Try to provide more context if possible
        return f"Failed to decode JSON from API response. Content: {response.text[:500]}...", 500 # Show start of response
    except Exception as e: # Catch other potential errors during request/json parsing
        return f"An unexpected error occurred during data fetching: {str(e)}", 500


    # --- Data Processing ---
    try:
        df = pd.DataFrame(data["data"], columns=["date", "value"])
        df.rename(columns={"date": "Date", "value": "DISCHARGE"}, inplace=True)
        df["DISCHARGE"] = pd.to_numeric(df["DISCHARGE"], errors='coerce')
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce') # Convert Date earlier

        # Drop rows where essential data is missing AFTER conversion attempts
        df = df.dropna(subset=['DISCHARGE', 'Date'])

        # Check if DataFrame is empty after cleaning
        if df.empty:
            return f"No valid discharge data points found for site {site_id} after cleaning.", 500

        metadata_fields = ["station_id", "station_name", "system_name", "units"]
        metadata = {field: data.get(field, "N/A") for field in metadata_fields}
        # No need to add metadata to df columns for plotting

        # --- Flagging Logic ---
        df['FLAG_NEGATIVE'] = (df['DISCHARGE'] < 0) & (df['DISCHARGE'] != 0)
        df['FLAG_ZERO'] = (df['DISCHARGE'] == 0)

        # Ensure percentile calculations don't fail on empty or all-zero data
        non_zero_discharge = df[df['DISCHARGE'] != 0]['DISCHARGE']
        if not non_zero_discharge.empty:
            discharge_95 = np.percentile(non_zero_discharge, 95)
            Q1, Q3 = non_zero_discharge.quantile([0.25, 0.75])
            IQR = Q3 - Q1

            df['FLAG_Discharge'] = (df['DISCHARGE'] > discharge_95) & (df['DISCHARGE'] != 0)
            df['FLAG_IQR'] = ((df['DISCHARGE'] < Q1 - 1.5 * IQR) | (df['DISCHARGE'] > Q3 + 1.5 * IQR)) & (df['DISCHARGE'] != 0)

            df['RATE_OF_CHANGE'] = df['DISCHARGE'].diff().abs()
            # Need to handle NaN in RoC for the first element
            df['FLAG_RoC'] = (df['RATE_OF_CHANGE'] > discharge_95) & (df['DISCHARGE'] != 0) & (~df['RATE_OF_CHANGE'].isna())

             # Isolation Forest only on non-zero values
            df_clean = df[df['DISCHARGE'] != 0].copy()
            if not df_clean.empty and 'DISCHARGE' in df_clean:
                 # Check if there's enough data for Isolation Forest
                 if len(df_clean[['DISCHARGE']].dropna()) >= 1:
                    model = IsolationForest(contamination='auto' if len(df_clean) > 1 else 0.0, random_state=42) # Use 'auto' contamination or handle single point case
                    df_clean['OUTLIER_IF'] = model.fit_predict(df_clean[['DISCHARGE']])
                    df['OUTLIER_IF'] = False
                    df.loc[df_clean.index, 'OUTLIER_IF'] = df_clean['OUTLIER_IF'] == -1
                 else:
                    df['OUTLIER_IF'] = False # Not enough data to run IF
            else:
                 df['OUTLIER_IF'] = False # No non-zero data
        else:
            # Handle case where all discharge values are zero or NaN
            df['FLAG_Discharge'] = False
            df['FLAG_IQR'] = False
            df['FLAG_RoC'] = False
            df['OUTLIER_IF'] = False


        # Calculate FLAG_REPEATED (handles non-zero constraint within logic)
        df['FLAG_REPEATED'] = (
            df['DISCHARGE']
            .where(df['DISCHARGE'] != 0) # Process only non-zeros
            .groupby((df['DISCHARGE'] != df['DISCHARGE'].shift()).cumsum()) # Group consecutive identical non-zeros
            .transform('size') >= 3 # Check if group size is 3 or more
        )
        # Ensure result is boolean and handle NaNs possibly introduced by where/transform if needed (fillna(False) might be safest)
        df['FLAG_REPEATED'] = df['FLAG_REPEATED'].fillna(False).astype(bool)


        # Determine overall flagged status (ensure all flag columns exist)
        flag_cols_for_any = [
             'FLAG_NEGATIVE', 'FLAG_ZERO', 'FLAG_REPEATED',
             'FLAG_IQR', 'OUTLIER_IF', 'FLAG_Discharge', 'FLAG_RoC'
        ]
        # Ensure all flag columns exist before using them, default to False if created conditionally
        for col in flag_cols_for_any:
            if col not in df.columns:
                df[col] = False
        df['FLAGGED'] = df[flag_cols_for_any].any(axis=1)

    except Exception as e:
         # Catch errors during pandas/sklearn processing
         import traceback
         tb_str = traceback.format_exc()
         return f"An error occurred during data processing: {str(e)}\nTraceback:\n{tb_str}", 500


    # --- Plotting ---
    fig = go.Figure()

    # Convert Date to string format suitable for Plotly AFTER all processing
    # Using ISO format is generally robust
    df["Date_str"] = df["Date"].dt.strftime('%Y-%m-%dT%H:%M:%S')

    # Add main discharge trace with count
    total_points = len(df)
    fig.add_trace(go.Scatter(
        x=df['Date_str'].tolist(),
        y=df['DISCHARGE'].tolist(),
        mode='lines',
        line=dict(color='lightgray', width=1.5),
        name=f'Mean Daily Discharge ({total_points})' # Add count here
    ))

    # Define flag colors and base names
    flag_colors = {
        'FLAG_NEGATIVE': ('red', 'Negative (-)'),
        'FLAG_ZERO': ('blue', 'Value = 0'),
        'FLAG_REPEATED': ('green', 'Repeated (≥3 days)'),
        'FLAG_RoC': ('brown', 'Rate of Change'),
        'FLAG_IQR': ('orange', 'IQR Outlier'),
        'OUTLIER_IF': ('teal', 'Isolation Forest'),
        'FLAG_Discharge': ('purple', 'Above 95th Percentile')
    }

    # Add flagged points traces with counts
    for flag, (color, base_name) in flag_colors.items():
         if flag in df.columns: # Ensure the flag column exists
            subset = df[df[flag]]
            count = len(subset) # Get count for this flag
            if not subset.empty:
                fig.add_trace(go.Scatter(
                    x=subset['Date_str'].tolist(),
                    y=subset['DISCHARGE'].tolist(),
                    mode='markers',
                    marker=dict(color=color, size=7),
                    name=f'{base_name} ({count})' # Add count to legend name
                ))

    # Update layout: move legend to top-center
    fig.update_layout(
        title=dict(text=f"Flagged Discharge Points - {metadata.get('station_name', f'Station {site_id}')}", x=0.5),
        yaxis_title=f"Mean Daily Discharge ({metadata.get('units', 'CFS')})", # Use units from metadata
        template="plotly_white",
        width=1400,
        height=700, # Consider making height adjustable or slightly larger if legend takes up space
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="bottom", # Anchor legend's bottom edge
            y=1.02,           # Position anchor just above the plot (adjust as needed)
            xanchor="center", # Anchor legend's center
            x=0.5             # Position anchor at the horizontal center of the plot
        )
    )

    # --- Rendering ---
    try:
        plot_json = fig.to_dict() # Use to_dict for render_template_string context
    except Exception as e:
        return f"Error converting Plotly figure to JSON: {str(e)}", 500


    return render_template_string(HTML_TEMPLATE, plot_json=plot_json, site_id=site_id)


if __name__ == '__main__':
    # Make sure to install waitress or gunicorn for production
    # Example using waitress (pip install waitress):
    # from waitress import serve
    # serve(app, host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))

    # Or run with gunicorn: gunicorn --bind 0.0.0.0:$PORT main:app

    # Development server (for local testing only):
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
