import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from flask import Flask, request, render_template_string
import plotly.graph_objects as go
import plotly.utils
import traceback
import json

app = Flask(__name__)

# (HTML_TEMPLATE remains the same)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Discharge Flags for Site {{ site_id }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        h2 { color: #333; }
        .error { color: red; font-weight: bold; }
        .warning { color: orange; }
        .nodata { color: gray; }
    </style>
</head>
<body>
    <h2>Discharge Data Flags for Site ID: {{ site_id }}</h2>

    {% if error %}
        <p class="error">Error: {{ error }}</p>
    {% elif warning %}
        <p class="warning">Warning: {{ warning }}</p>
        {% if plot_json_str %} {# Check if plot json exists for warning case #}
            <div id="plot"></div>
        {% endif %}
    {% elif nodata %}
         <p class="nodata">{{ nodata }}</p>
    {% elif plot_json_str %} {# Check if plot json exists for success case #}
        <div id="plot"></div>
    {% else %}
        <p class="error">An unknown issue occurred and the plot could not be generated.</p>
    {% endif %}

    <script>
        var plot_json_str = {{ plot_json_str | default('null') | tojson | safe }}; // Pass as string
        if (plot_json_str) {
            try {
                var plot_json = JSON.parse(plot_json_str); // Parse JSON string
                Plotly.newPlot('plot', plot_json.data, plot_json.layout, {responsive: true});
            } catch (e) {
                console.error("Plotly error:", e);
                document.getElementById('plot').innerHTML = '<p class="error">Failed to render plot. Check browser console. Error: ' + e + '</p>';
                // Optionally display the raw JSON string for debugging
                // document.getElementById('plot').innerHTML += '<pre>' + plot_json_str + '</pre>';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/plot')
def plot_site():
    site_id = request.args.get("id")
    plot_json_str = None
    error_message = None
    warning_message = None
    nodata_message = None

    if not site_id:
        error_message = "Missing 'id' parameter in query string."
        return render_template_string(HTML_TEMPLATE, error=error_message, site_id="N/A", plot_json_str=None), 400

    # --- Data Fetching ---
    try:
        end_date = datetime.today().strftime("%Y-%m-%d")
        api_url = f"https://www.waterrights.utah.gov/dvrtdb/daily-chart.asp?station_id={site_id}&end_date={end_date}&f=json"
        print(f"Fetching data for site {site_id}: {api_url}")
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "data" not in data or not isinstance(data["data"], list):
            error_message = f"API response 'data' field is invalid or missing for site {site_id}. Response: {response.text[:500]}"
            return render_template_string(HTML_TEMPLATE, error=error_message, site_id=site_id, plot_json_str=None), 502

    except requests.exceptions.RequestException as e:
        tb_str = traceback.format_exc(); print(tb_str)
        error_message = f"Error fetching data from API: {str(e)}"
        return render_template_string(HTML_TEMPLATE, error=error_message, site_id=site_id, plot_json_str=None), 500
    except Exception as e:
        tb_str = traceback.format_exc(); print(tb_str)
        error_message = f"An unexpected error occurred during data fetching: {str(e)}"
        return render_template_string(HTML_TEMPLATE, error=error_message, site_id=site_id, plot_json_str=None), 500

    # --- Data Processing & Plotting ---
    try:
        if not data["data"]:
            nodata_message = f"No time series data returned from API for site {site_id} for the requested period."
            return render_template_string(HTML_TEMPLATE, nodata=nodata_message, site_id=site_id, plot_json_str=None), 200
        else:
            df = pd.DataFrame(data["data"], columns=["date", "value"])
            df.rename(columns={"date": "Date", "value": "DISCHARGE"}, inplace=True)

            metadata_fields = ["station_id", "station_name", "system_name", "units"]
            metadata = {field: data.get(field, "N/A") for field in metadata_fields}
            for key, value in metadata.items():
                 df[key] = value

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['DISCHARGE'] = pd.to_numeric(df['DISCHARGE'], errors='coerce')

            # Initial sort by Date
            df = df.dropna(subset=['Date', 'DISCHARGE']).sort_values(by='Date').reset_index(drop=True)

            if df.empty:
                nodata_message = f"No valid, plottable data points found for site {site_id} after cleaning."
                return render_template_string(HTML_TEMPLATE, nodata=nodata_message, site_id=site_id, plot_json_str=None), 200

            core_cols = ["Date", "DISCHARGE"]
            meta_cols = [col for col in metadata_fields if col in df.columns]
            all_cols = core_cols + meta_cols
            df = df[all_cols]

            # === FLAGGING CRITERIA === #
            df['FLAG_NEGATIVE'] = (df['DISCHARGE'] < 0) & (df['DISCHARGE'] != 0)
            df['FLAG_ZERO'] = (df['DISCHARGE'] == 0)

            df_nonzero = df[df["DISCHARGE"] > 0].copy()

            df["RATE_OF_CHANGE"] = np.nan
            df["OUTLIER_IF"] = False
            df["FLAG_REPEATED"] = False
            df["PERCENT_DEV"] = np.nan
            df["FLAG_RSD"] = False
            df["FLAG_Discharge"] = False
            df["FLAG_IQR"] = False
            df["FLAG_RoC"] = False

            discharge_90th_percentile = np.nan
            discharge_95th_percentile = np.nan
            Q1, Q3, IQR = np.nan, np.nan, np.nan

            if not df_nonzero.empty:
                Q1, Q3 = df_nonzero["DISCHARGE"].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                # Ensure dropna() before percentile calculation
                df_nonzero_dropna = df_nonzero["DISCHARGE"].dropna()
                if not df_nonzero_dropna.empty:
                    discharge_90th_percentile = np.percentile(df_nonzero_dropna, 90)
                    discharge_95th_percentile = np.percentile(df_nonzero_dropna, 95)

                # Sort subset by date for diff calculation
                df_nonzero = df_nonzero.sort_values(by="Date")
                df_nonzero["RATE_OF_CHANGE"] = df_nonzero["DISCHARGE"].diff().abs()
                df.update(df_nonzero[["RATE_OF_CHANGE"]]) # Update based on index

                # Repeated flag calculation
                non_zero_mask = df["DISCHARGE"] > 0
                if non_zero_mask.any():
                     df_sorted_nonzero = df.loc[non_zero_mask].sort_values('Date')
                     is_different = (df_sorted_nonzero["DISCHARGE"] != df_sorted_nonzero["DISCHARGE"].shift())
                     group_sizes = df_sorted_nonzero["DISCHARGE"].groupby(is_different.cumsum()).transform("size")
                     flag_repeated_temp = group_sizes >= 4
                     df.loc[df_sorted_nonzero.index, "FLAG_REPEATED"] = flag_repeated_temp

                # Isolation Forest
                df_iforest = df_nonzero[["DISCHARGE"]].dropna()
                if not df_iforest.empty and df_iforest['DISCHARGE'].nunique() > 1:
                    model = IsolationForest(contamination=0.05, random_state=42)
                    df_nonzero.loc[df_iforest.index, "OUTLIER_IF_PREDICT"] = model.fit_predict(df_iforest) # Use loc for alignment
                    df_nonzero["OUTLIER_IF"] = df_nonzero["OUTLIER_IF_PREDICT"] == -1
                    df.update(df_nonzero[["OUTLIER_IF"]])
                    df["OUTLIER_IF"] = df["OUTLIER_IF"].fillna(False).astype(bool)

                # RSD calculation
                mean_discharge = df_nonzero["DISCHARGE"].mean()
                if mean_discharge != 0:
                    df["PERCENT_DEV"] = ((df["DISCHARGE"] - mean_discharge).abs() / mean_discharge) * 100
                threshold = 1000
                df["FLAG_RSD"] = (df["PERCENT_DEV"] > threshold) & (df["DISCHARGE"] != 0) & df["PERCENT_DEV"].notna()
                df["FLAG_RSD"] = df["FLAG_RSD"].fillna(False)

                # --- Calculate flags based on thresholds ---
                if pd.notna(discharge_95th_percentile):
                    df["FLAG_Discharge"] = (df["DISCHARGE"] > discharge_95th_percentile) & (df["DISCHARGE"] > 0)
                if pd.notna(Q1) and pd.notna(Q3) and pd.notna(IQR):
                    if IQR > 0:
                        df["FLAG_IQR"] = ((df["DISCHARGE"] < Q1 - 1.5 * IQR) | (df["DISCHARGE"] > Q3 + 1.5 * IQR)) & (df["DISCHARGE"] > 0)
                    elif IQR == 0:
                        df["FLAG_IQR"] = (df["DISCHARGE"] != Q1) & (df["DISCHARGE"] > 0)
                if pd.notna(discharge_90th_percentile):
                    df["FLAG_RoC"] = (df["RATE_OF_CHANGE"] > discharge_90th_percentile) & df["RATE_OF_CHANGE"].notna()

            # === Ensure all flag columns exist and are boolean === #
            flag_cols_final = [
                "FLAG_NEGATIVE", "FLAG_ZERO", "FLAG_REPEATED", "OUTLIER_IF",
                "FLAG_RSD", "FLAG_Discharge", "FLAG_IQR", "FLAG_RoC"
            ]
            for col in flag_cols_final:
                if col not in df.columns:
                    df[col] = False
                else:
                    df[col] = df[col].fillna(False).astype(bool)

            # === Combined Flags === #
            df["FLAG_ABOVE_MAX_OVERLAP"] = (df["FLAG_IQR"] & df["FLAG_Discharge"] & df["OUTLIER_IF"]) & (df["DISCHARGE"] > 0)
            df["FLAG_LARGE_SPIKES"] = df["FLAG_RSD"] & df["FLAG_RoC"]

            # === Overall Flagged Indicator === #
            flags_for_overall = ["FLAG_NEGATIVE", "FLAG_ZERO", "FLAG_REPEATED", "FLAG_ABOVE_MAX_OVERLAP", "FLAG_LARGE_SPIKES"]
            df["FLAGGED"] = df[flags_for_overall].any(axis=1)

            # <<< --- ADDED LINE: Ensure final DataFrame is sorted by Date before plotting --- >>>
            df = df.sort_values(by='Date').reset_index(drop=True)

            # --- Plotting Setup ---
            plot_title = f"Flagged Data Points for {metadata.get('station_name', 'Station ' + site_id)}"
            units = metadata.get('units', 'CFS')

            flag_colors = {
                'FLAG_NEGATIVE': ('red', 'Negative (-)'),
                'FLAG_ZERO': ('blue', 'Value = 0'),
                'FLAG_REPEATED': ('green', 'Repeated (≥4 days)'),
                'FLAG_ABOVE_MAX_OVERLAP': ('brown', 'Above Suspected Max'),
                'FLAG_LARGE_SPIKES': ('orange', 'Large Spikes'),
            }

            fig = go.Figure() # Create the figure AFTER the final sort

            # Background Line
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df['DISCHARGE'], # Use the re-sorted df
                mode='lines',
                line=dict(color='lightgray', width=1.5),
                name='Mean Daily Discharge',
                connectgaps=False
            ))

            # Add flagged points
            any_flags_plotted = False
            for flag, (color, legend_name) in flag_colors.items():
                 subset = df[df[flag]] # Use the re-sorted df
                 if not subset.empty:
                     any_flags_plotted = True
                     fig.add_trace(go.Scatter(
                         x=subset['Date'], y=subset['DISCHARGE'],
                         mode='markers',
                         marker=dict(color=color, size=7),
                         name=legend_name
                     ))

            # Add horizontal line
            if "FLAG_ABOVE_MAX_OVERLAP" in df.columns:
                above_max_subset = df[df["FLAG_ABOVE_MAX_OVERLAP"]]
                if not above_max_subset.empty:
                    min_above_max = above_max_subset["DISCHARGE"].min()
                    if pd.notna(min_above_max) and np.isfinite(min_above_max):
                        fig.add_trace(go.Scatter(
                            # Use min/max dates from the final re-sorted df
                            x=[df["Date"].min(), df["Date"].max()],
                            y=[min_above_max, min_above_max],
                            mode="lines",
                            line=dict(color="brown", width=2, dash="dash"),
                            name="Min Value (Above Suspected Max)"
                        ))

            # Update layout
            fig.update_layout(
                title=dict(text=plot_title, x=0.5, font=dict(size=20)),
                xaxis=dict(title="Date", title_font=dict(size=18), tickfont=dict(size=14)),
                yaxis=dict(title=f"Mean Daily Discharge ({units})", title_font=dict(size=18), tickfont=dict(size=14)),
                legend=dict(
                    orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5,
                    title=dict(text="Flagging Criteria:", font=dict(size=16)), font=dict(size=14)
                ),
                template="plotly_white",
                width=1400, height=700,
                margin=dict(t=80, b=120)
            )

            if not any_flags_plotted and not nodata_message:
                warning_message = f"Data processed successfully for site {site_id}, but no data points met the specific flagging criteria being plotted."

            # Convert figure to JSON string
            plot_json_str = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    except Exception as e:
        tb_str = traceback.format_exc(); print(tb_str)
        error_message = f"An error occurred during data processing or plot generation: {str(e)}"
        return render_template_string(HTML_TEMPLATE, error=error_message, site_id=site_id, plot_json_str=None), 500

    # --- Rendering ---
    return render_template_string(HTML_TEMPLATE, plot_json_str=plot_json_str, site_id=site_id, error=error_message, warning=warning_message, nodata=nodata_message)

# --- Main execution block ---
if __name__ == '__main__':
    print("Starting Flask development server...")
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8080))
    print(f"Access the plot via http://{host}:{port}/plot?id=YOUR_SITE_ID (e.g., http://localhost:{port}/plot?id=10987)")
    debug_mode = os.environ.get("FLASK_DEBUG", "True").lower() == "true"
    app.run(debug=debug_mode, host=host, port=port)
