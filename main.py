import os
from flask import Flask, request, render_template_string
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import plotly.io as pio
import json

app = Flask(__name__)

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
    site_id = request.args.get('id')
    if not site_id:
        return "Error: No site ID provided. Use ?id=5 in the URL.", 400
    
    end_date = datetime.today().strftime("%Y-%m-%d")
    api_url = f"https://www.waterrights.utah.gov/dvrtdb/daily-chart.asp?station_id={site_id}&end_date={end_date}&f=json"
    response = requests.get(api_url)
    if response.status_code != 200:
        return f"Error fetching data: {response.status_code}", 400
    
    data = response.json()
    if "data" not in data:
        return "Error: No 'data' key found in API response.", 400
    
    df = pd.DataFrame(data["data"], columns=["date", "value"])
    df.rename(columns={"date": "Date", "value": "DISCHARGE"}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(by='Date', ascending=True, ignore_index=True)  # Explicit sorting
    
    print("Sorted dates (first 10):", df['Date'].tolist()[:10])  # Debugging
    
    metadata_fields = ["station_id", "station_name", "system_name", "units"]
    metadata = {field: data.get(field, "N/A") for field in metadata_fields}
    for key, value in metadata.items():
        df[key] = value
    
    df['DISCHARGE'] = pd.to_numeric(df['DISCHARGE'], errors='coerce')
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S')  # Convert to ISO format
    
    flag_colors = {
        'FLAG_NEGATIVE': ('red', 'Negative (-)'),
        'FLAG_ZERO': ('blue', 'Value = 0'),
    }
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['DISCHARGE'], mode='lines', line=dict(color='lightgray', width=1.5), name='Mean Daily Discharge'))
    for flag, (color, legend_name) in flag_colors.items():
        subset = df[df[flag]]
        if not subset.empty:
            fig.add_trace(go.Scatter(x=subset['Date'], y=subset['DISCHARGE'], mode='markers', marker=dict(color=color, size=7), name=legend_name))
    fig.update_layout(title=f"Flagged Data Points for {metadata.get('station_name', 'Station ' + site_id)}", width=1400, height=700)
    plot_json = json.dumps(fig, default=str)  # Ensure JSON serialization is correct
    return render_template_string(HTML_TEMPLATE, plot_json=plot_json, site_id=site_id)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
