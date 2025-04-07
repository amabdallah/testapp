from flask import Flask, request, jsonify
from datetime import datetime
import requests
import pandas as pd
import plotly.graph_objects as go
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/plot')
def generate_plot():
    site_id = request.args.get('id')
    if not site_id:
        return jsonify({"error": "No site ID provided. Use ?id=5 in the URL."}), 400

    end_date = datetime.today().strftime("%Y-%m-%d")
    api_url = f"https://www.waterrights.utah.gov/dvrtdb/daily-chart.asp?station_id={site_id}&end_date={end_date}&f=json"
    logging.info(f"Fetching data for site ID {site_id} from {api_url}")

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()

        try:
            data = response.json()
        except Exception as e:
            logging.error("Failed to parse JSON from response: %s", e)
            return jsonify({"error": "Failed to parse response from external API."}), 500

    except requests.exceptions.Timeout:
        logging.error("Timeout occurred when calling %s", api_url)
        return jsonify({"error": "The data source took too long to respond. Please try again later."}), 504

    except requests.exceptions.RequestException as e:
        logging.error("Request failed: %s", e)
        return jsonify({"error": "Error retrieving data from the external source."}), 502

    if "data" not in data or not data["data"]:
        logging.warning("API response missing 'data' or is empty for site %s", site_id)
        return jsonify({"error": "No discharge data available for this site."}), 404

    try:
        df = pd.DataFrame(data["data"], columns=["date", "value"])
        df.rename(columns={"date": "Date", "value": "DISCHARGE"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df = df.sort_values(by="Date", ascending=True, ignore_index=True)
        df["DISCHARGE"] = pd.to_numeric(df["DISCHARGE"], errors='coerce')

        # Optional: Handle flags safely
        flag_colors = {
            'FLAG_NEGATIVE': ('red', 'Negative (-)'),
            'FLAG_ZERO': ('blue', 'Value = 0'),
        }

        metadata_fields = ["station_id", "station_name", "system_name", "units"]
        metadata = {field: data.get(field, "N/A") for field in metadata_fields}
        for key, value in metadata.items():
            df[key] = value

        df["Date"] = df["Date"].dt.strftime('%Y-%m-%dT%H:%M:%S')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["DISCHARGE"],
            mode="lines",
            line=dict(color="gray", width=1.5),
            name="Mean Daily Discharge"
        ))

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
            title=f"Discharge Data for {metadata.get('station_name', 'Station ' + site_id)}",
            xaxis_title="Date",
            yaxis_title=metadata.get("units", "Discharge Value"),
            width=800,
            height=600
        )

        plot_html = fig.to_html(full_html=False)
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Discharge Plot</title>
        </head>
        <body>
            <h1>Discharge Data for {metadata.get('station_name', 'Station ' + site_id)}</h1>
            {plot_html}
            <p>Station ID: {metadata.get('station_id', 'N/A')}</p>
            <p>System Name: {metadata.get('system_name', 'N/A')}</p>
            <p>Units: {metadata.get('units', 'N/A')}</p>
        </body>
        </html>
        """

    except Exception as e:
        logging.error("Error processing data and generating plot: %s", e)
        return jsonify({"error": "Error processing data and generating the plot."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int("8080"))
