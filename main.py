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
            return "No 'data' key in API response", 500
    except Exception as e:
        return f"API error: {str(e)}", 500

    df = pd.DataFrame(data["data"], columns=["date", "value"])
    df.rename(columns={"date": "Date", "value": "DISCHARGE"}, inplace=True)
    df["DISCHARGE"] = pd.to_numeric(df["DISCHARGE"], errors='coerce')

    metadata_fields = ["station_id", "station_name", "system_name", "units"]
    metadata = {field: data.get(field, "N/A") for field in metadata_fields}
    for k, v in metadata.items():
        df[k] = v

    df['FLAG_NEGATIVE'] = (df['DISCHARGE'] < 0) & (df['DISCHARGE'] != 0)
    df['FLAG_ZERO'] = (df['DISCHARGE'] == 0)

    discharge_95 = np.percentile(df[df['DISCHARGE'] != 0]['DISCHARGE'].dropna(), 95)
    df['FLAG_Discharge'] = (df['DISCHARGE'] > discharge_95) & (df['DISCHARGE'] != 0)

    Q1, Q3 = df[df['DISCHARGE'] != 0]['DISCHARGE'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df['FLAG_IQR'] = ((df['DISCHARGE'] < Q1 - 1.5 * IQR) | (df['DISCHARGE'] > Q3 + 1.5 * IQR)) & (df['DISCHARGE'] != 0)

    df['RATE_OF_CHANGE'] = df['DISCHARGE'].diff().abs()
    df['FLAG_RoC'] = (df['RATE_OF_CHANGE'] > discharge_95) & (df['DISCHARGE'] != 0)

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

    flag_colors = {
        'FLAG_NEGATIVE': ('red', 'Negative (-)'),
        'FLAG_ZERO': ('blue', 'Value = 0'),
        'FLAG_REPEATED': ('green', 'Repeated (≥3 days)'),
        'FLAG_RoC': ('brown', 'Rate of Change'),
        'FLAG_IQR': ('orange', 'IQR Outlier'),
        'OUTLIER_IF': ('teal', 'Isolation Forest'),
        'FLAG_Discharge': ('purple', 'Above 95th Percentile')
    }

    fig = go.Figure()

    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])
    df["Date"] = df["Date"].dt.strftime('%Y-%m-%dT%H:%M:%S')

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['DISCHARGE'],
        mode='lines', line=dict(color='lightgray', width=1.5),
        name='Mean Daily Discharge'
    ))

    for flag, (color, name) in flag_colors.items():
        subset = df[df[flag]]
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset['Date'], y=subset['DISCHARGE'],
                mode='markers', marker=dict(color=color, size=7),
                name=name
            ))

    fig.update_layout(
        title=dict(text=f"Flagged Points - {metadata.get('station_name', f'Station {site_id}')}", x=0.5),
        yaxis_title="Mean Daily Discharge (CFS)",
        template="plotly_white",
        width=1400,
        height=700,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
    )

    return render_template_string(HTML_TEMPLATE, plot_json=fig.to_dict(), site_id=site_id)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
