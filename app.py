from flask import Flask, request, render_template_string
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)

# HTML template with Plotly
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Discharge Data Plot</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h2>Discharge Data for Site ID: {{ site_id }}</h2>
    <div id="plot"></div>
    <script>
        var plot_data = {{ plot_json | safe }};
        Plotly.newPlot('plot', plot_data.data, plot_data.layout);
    </script>
</body>
</html>
"""

@app.route('/plot')
def generate_plot():
    # Get site_ID from the URL
    site_id = request.args.get('id')
    
    if not site_id:
        return "Error: No site ID provided. Use ?id=5 in the URL.", 400

    # Fetch data from API
    end_date = datetime.today().strftime("%Y-%m-%d")
    api_url = f"https://www.waterrights.utah.gov/dvrtdb/daily-chart.asp?station_id={site_id}&end_date={end_date}&f=json"

    response = requests.get(api_url)

    if response.status_code != 200:
        return f"Error fetching data: {response.status_code}", 400
    
    data = response.json()

    if "data" not in data:
        return "Error: No 'data' key found in API response.", 400

    # Convert data to DataFrame
    df = pd.DataFrame(data["data"], columns=["date", "value"])
    df.rename(columns={"date": "Date", "value": "DISCHARGE"}, inplace=True)

    # Convert 'Date' to datetime and sort
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(by='Date')

    # Metadata
    metadata_fields = ["station_id", "station_name", "system_name", "units"]
    metadata = {field: data.get(field, "N/A") for field in metadata_fields}

    for key, value in metadata.items():
        df[key] = value

    df['DISCHARGE'] = pd.to_numeric(df['DISCHARGE'], errors='coerce')

    # === FLAGGING CRITERIA === #
    df['FLAG_NEGATIVE'] = (df['DISCHARGE'] < 0) & (df['DISCHARGE'] != 0)
    df['FLAG_ZERO'] = (df['DISCHARGE'] == 0)

    discharge_95th_percentile = np.percentile(df[df['DISCHARGE'] != 0]['DISCHARGE'].dropna(), 95)
    df['FLAG_Discharge'] = (df['DISCHARGE'] > discharge_95th_percentile) & (df['DISCHARGE'] != 0)

    Q1, Q3 = df[df['DISCHARGE'] != 0]['DISCHARGE'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df['FLAG_IQR'] = ((df['DISCHARGE'] < lower_bound) | (df['DISCHARGE'] > upper_bound)) & (df['DISCHARGE'] != 0)

    df['RATE_OF_CHANGE'] = df['DISCHARGE'].diff().abs()
    df['FLAG_RoC'] = (df['RATE_OF_CHANGE'] > discharge_95th_percentile) & (df['DISCHARGE'] != 0)

    df['FLAG_REPEATED'] = (
        df['DISCHARGE']
        .where(df['DISCHARGE'] != 0)
        .groupby((df['DISCHARGE'] != df['DISCHARGE'].shift()).cumsum())
        .transform('count') >= 3
    )

    df_clean = df[df['DISCHARGE'] != 0].dropna(subset=['DISCHARGE'])
    model = IsolationForest(contamination=0.05, random_state=42)
    df_clean['OUTLIER_IF'] = model.fit_predict(df_clean[['DISCHARGE']])
    df['OUTLIER_IF'] = False
    df.loc[df_clean.index, 'OUTLIER_IF'] = df_clean['OUTLIER_IF'] == -1

    df['FLAGGED'] = df[
        ['FLAG_NEGATIVE', 'FLAG_ZERO', 'FLAG_REPEATED', 'FLAG_IQR', 'OUTLIER_IF', 'FLAG_Discharge', 'FLAG_RoC']
    ].any(axis=1)

    plot_title = f"Flagged Data Points for {metadata.get('station_name', 'Station ' + site_id)}"

    flag_colors = {
        'FLAG_NEGATIVE': ('red', 'Negative (-)'),
        'FLAG_ZERO': ('blue', 'Value = 0'),
        'FLAG_REPEATED': ('green', 'Repeated (≥3 days)'),
        'FLAG_RoC': ('brown', 'Rate of Change - Outlier'),
        'FLAG_IQR': ('orange', 'IQR Test - Outlier'),
        'OUTLIER_IF': ('teal', 'Isolation Forest - Outlier'),
        'FLAG_Discharge': ('purple', 'Above the 95th % Percentile')
    }

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['DISCHARGE'],
        mode='lines',
        line=dict(color='lightgray', width=1.5),
        name='Mean Daily Discharge'
    ))

    for flag, (color, legend_name) in flag_colors.items():
        subset = df[df[flag]]
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset['Date'], y=subset['DISCHARGE'],
                mode='markers',
                marker=dict(color=color, size=7),
                name=legend_name
            ))

    fig.update_layout(
        title=plot_title,
        xaxis=dict(type='date'),  # Force x-axis to be time-series
        yaxis_title="Mean Daily Discharge (CFS)",
        legend=dict(orientation="h", y=-0.2, x=0.5),
        template="plotly_white",
        width=1400,
        height=700
    )

    plot_json = pio.to_json(fig)

    return render_template_string(HTML_TEMPLATE, plot_json=plot_json, site_id=site_id)

if __name__ == '__main__':
    app.run(debug=True)
