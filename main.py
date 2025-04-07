import os
import json
import logging
from flask import Flask, request, render_template_string
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Initialize Flask app and logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

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
        var plot_data = {{ plot_json | tojson | safe }};
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
    logging.info(f"Fetching data from: {api_url}")

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        logging.error("API request timed out.")
        return "The data source took too long to respond. Please try again later.", 504
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return "Error retrieving data from the external source.", 502
    except Exception as e:
        logging.error(f"Error parsing JSON: {e}")
        return "Failed to parse response from external API.", 500

    if "data" not in data or not data["data"]:
        logging.warning(f"No data found in API response for site ID {site_id}")
        return "No discharge data available for this site.", 404

    try:
        # Build DataFrame
        df = pd.DataFrame(data["data"], columns=["date", "value"])
        df.rename(columns={"date": "Date", "value": "DISCHARGE"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df = df.sort_values(by="Date", ascending=True, ignore_index=True)
        df["DISCHARGE"] = pd.to_numeric(df["DISCHARGE"], errors='coerce')

        # Log column names for debugging
        logging.info(f"Data columns: {df.columns.tolist()}")

        # Add metadata fields
        metadata_fields = ["station_id", "station_name", "system_name", "units"]
        metadata = {field: data.get(field, "N/A") for field in metadata_fields}
        for key, value in metadata.items():
            df[key] = value

        # Format date for plot
        df["Date"] = df["Date"].dt.strftime('%Y-%m-%dT%H:%M:%S')

        # Start building plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["DISCHARGE"],
            mode="lines",
            line=dict(color="gray", width=1.5),
            name="Mean Daily Discharge"
        ))

        # Optional: FLAG handling
        flag_colors = {
            'FLAG_NEGATIVE': ('red', 'Negative (-)'),
            'FLAG_ZERO': ('blue', 'Value = 0'),
        }

        for flag, (color, label) in flag_colors.items():
            if flag in df.columns:
                subset = df[df[flag]]
                if not subset.empty:
                    fig.add_trace(go.Scatter(
                        x=subset["Date"],
                        y=subset["DISCHARGE"],
                        mode="markers",
                        marker=dict(color=color, size=7),
                        name=label
                    ))

        fig.update_layout(
            title=f"Flagged Data Points for {metadata.get('station_name', 'Station ' + site_id)}",
            width=1400,
            height=700
        )

        return render_template_string(HTML_TEMPLATE, plot_json=fig.to_dict(), site_id=site_id)

    except Exception as e:
        logging.exception("Unexpected error while processing data")
        return "An error occurred while generating the plot.", 500


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
