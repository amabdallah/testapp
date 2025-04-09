import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from flask import Flask, request, render_template_string
import plotly.graph_objects as go
import traceback

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
        {% if plot_json %}
            <div id="plot"></div>
        {% endif %}
    {% elif nodata %}
         <p class="nodata">{{ nodata }}</p>
    {% elif plot_json %}
        <div id="plot"></div>
    {% else %}
        <p class="error">An unknown issue occurred and the plot could not be generated.</p>
    {% endif %}

    <script>
        var plot_json = {{ plot_json | default('null') | tojson | safe }};
        if (plot_json) {
            try {
                Plotly.newPlot('plot', plot_json.data, plot_json.layout, {responsive: true});
            } catch (e) {
                console.error("Plotly error:", e);
                document.getElementById('plot').innerHTML = '<p class="error">Failed to render plot. Check browser console.</p>';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/plot')
def plot_site():
    site_id = request.args.get("id")
    plot_json = None
    error_message = None
    warning_message = None
    nodata_message = None

    if not site_id:
        error_message = "Missing 'id' parameter in query string."
        return render_template_string(HTML_TEMPLATE, error=error_message, site_id="N/A"), 400

    # --- Data Fetching ---
    # (Error handling for fetch remains the same)
    try:
        end_date = datetime.today().strftime("%Y-%m-%d")
        api_url = f"https://www.waterrights.utah.gov/dvrtdb/daily-chart.asp?station_id={site_id}&end_date={end_date}&f=json"
        print(f"Fetching data for site {site_id}: {api_url}")
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "data" not in data or not isinstance(data["data"], list):
             error_message = f"API response 'data' field invalid or missing for site {site_id}."
             return render_template_string(HTML_TEMPLATE, error=error_message, site_id=site_id), 502
    except Exception as e: # Catch all fetch errors here
        # (Consolidated error handling for fetch)
        tb_str = traceback.format_exc(); print(tb_str)
        error_message = f"Error fetching data: {str(e)}"
        return render_template_string(HTML_TEMPLATE, error=error_message, site_id=site_id), 500


    # --- Data Processing & Plotting ---
    try:
        if not data["data"]:
            nodata_message = f"No time series data returned from API for site {site_id}."
            return render_template_string(HTML_TEMPLATE, nodata=nodata_message, site_id=site_id), 200
        else:
            df = pd.DataFrame(data["data"], columns=["date", "value"])
            df.rename(columns={"date": "Date", "value": "DISCHARGE"}, inplace=True)

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['DISCHARGE'] = pd.to_numeric(df['DISCHARGE'], errors='coerce')
        df = df.dropna(subset=['Date', 'DISCHARGE']).sort_values(by='Date').reset_index(drop=True)

        if df.empty:
            nodata_message = f"No valid, plottable data points found for site {site_id} after cleaning."
            return render_template_string(HTML_TEMPLATE, nodata=nodata_message, site_id=site_id), 200

        metadata_fields = ["station_id", "station_name", "system_name", "units"]
        metadata = {field: data.get(field, f"N/A") for field in metadata_fields}
        station_name = metadata.get('station_name', f'Station {site_id}')
        units = metadata.get('units', 'Unknown Units')

        # 1. Define Seasons and Segment Data (unchanged)
        irrigation_months = [4, 5, 6, 7, 8, 9]
        df['Season'] = df['Date'].dt.month.apply(lambda month: 'Irrigation' if month in irrigation_months else 'Non-Irrigation')
        df['Discharge_Irrigation'] = df.apply(lambda row: row['DISCHARGE'] if row['Season'] == 'Irrigation' else np.nan, axis=1)
        df['Discharge_NonIrrigation'] = df.apply(lambda row: row['DISCHARGE'] if row['Season'] == 'Non-Irrigation' else np.nan, axis=1)

        # 2. Flagging Criteria (logic unchanged, condensed for brevity)
        # ... (full flagging logic as before) ...
        df['FLAG_NEGATIVE'] = (df['DISCHARGE'] < 0) & (df['DISCHARGE'].notna()) & (df['DISCHARGE'] != 0)
        df['FLAG_ZERO'] = (df['DISCHARGE'] == 0) & (df['DISCHARGE'].notna())
        non_zero_discharge = df[(df['DISCHARGE'].notna()) & (df['DISCHARGE'] != 0)]['DISCHARGE']
        if not non_zero_discharge.empty:
            discharge_95th_percentile = np.percentile(non_zero_discharge, 95)
            df['FLAG_Discharge'] = (df['DISCHARGE'] > discharge_95th_percentile) & (df['DISCHARGE'].notna()) & (df['DISCHARGE'] != 0)
            Q1, Q3 = non_zero_discharge.quantile([0.25, 0.75]); IQR = Q3 - Q1
            if IQR > 0: lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR; df['FLAG_IQR'] = ((df['DISCHARGE'] < lower_bound) | (df['DISCHARGE'] > upper_bound)) & (df['DISCHARGE'].notna()) & (df['DISCHARGE'] != 0)
            else: df['FLAG_IQR'] = (df['DISCHARGE'] != Q1) & (df['DISCHARGE'].notna()) & (df['DISCHARGE'] != 0)
            df['RATE_OF_CHANGE'] = df['DISCHARGE'].diff().abs()
            if 'discharge_95th_percentile' in locals(): df['FLAG_RoC'] = (df['RATE_OF_CHANGE'] > discharge_95th_percentile) & (df['DISCHARGE'].notna()) & (df['DISCHARGE'] != 0) & (df['RATE_OF_CHANGE'].notna())
            else: df['FLAG_RoC'] = False
            non_zero_mask = df['DISCHARGE'].notna() & (df['DISCHARGE'] != 0); df['FLAG_REPEATED'] = False
            if non_zero_mask.any(): groups = (df.loc[non_zero_mask, 'DISCHARGE'] != df.loc[non_zero_mask, 'DISCHARGE'].shift()).cumsum(); group_sizes = df.loc[non_zero_mask, 'DISCHARGE'].groupby(groups).transform('size'); df.loc[non_zero_mask, 'FLAG_REPEATED'] = (group_sizes >= 3)
            df_clean = df[(df['DISCHARGE'].notna()) & (df['DISCHARGE'] != 0)].copy()
            if not df_clean.empty and df_clean['DISCHARGE'].nunique() > 1: model = IsolationForest(contamination='auto', random_state=42); df_clean['OUTLIER_IF_PREDICT'] = model.fit_predict(df_clean[['DISCHARGE']]); df['OUTLIER_IF'] = False; df.loc[df_clean.index, 'OUTLIER_IF'] = (df_clean['OUTLIER_IF_PREDICT'] == -1)
            else: df['OUTLIER_IF'] = False
            mean_discharge = non_zero_discharge.mean()
            if mean_discharge != 0: df['PERCENT_DEV'] = ((df['DISCHARGE'] - mean_discharge).abs() / mean_discharge) * 100; threshold = 1000; df['FLAG_RSD'] = (df['PERCENT_DEV'] > threshold) & (df['DISCHARGE'].notna()) & (df['DISCHARGE'] != 0) & (df['PERCENT_DEV'].notna())
            else: df['PERCENT_DEV'] = np.nan; df['FLAG_RSD'] = False
        else: [df.update({f: False}) for f in ['FLAG_Discharge', 'FLAG_IQR', 'FLAG_RoC', 'FLAG_REPEATED', 'OUTLIER_IF', 'FLAG_RSD']]
        flag_cols_to_check = ['FLAG_NEGATIVE', 'FLAG_ZERO', 'FLAG_REPEATED', 'FLAG_IQR', 'OUTLIER_IF', 'FLAG_Discharge', 'FLAG_RoC', 'FLAG_RSD']; [df.update({col: False}) for col in flag_cols_to_check if col not in df.columns]; df['FLAGGED'] = df[flag_cols_to_check].any(axis=1)

        # 3. Create Plot and Add Traces (unchanged)
        plot_title = f"Flagged Data Points & Discharge by Season for {station_name}"
        flag_colors = { 'FLAG_NEGATIVE': ('red', 'Negative (-)'), 'FLAG_ZERO': ('blue', 'Value = 0'), 'FLAG_REPEATED': ('green', 'Repeated (≥3)'), 'FLAG_RoC': ('brown', 'RoC Outlier'), 'FLAG_IQR': ('orange', 'IQR Outlier'), 'OUTLIER_IF': ('teal', 'IF Outlier'), 'FLAG_Discharge': ('purple', '> 95th Perc.'), 'FLAG_RSD': ('magenta', 'RSD Outlier') }
        fig = go.Figure()
        line_width = 2.0
        fig.add_trace(go.Scatter( x=df['Date'].tolist(), y=df['Discharge_Irrigation'].tolist(), mode='lines', line=dict(color='lightgreen', width=line_width), name='Irrigation Season Discharge', connectgaps=False, showlegend=True ))
        fig.add_trace(go.Scatter( x=df['Date'].tolist(), y=df['Discharge_NonIrrigation'].tolist(), mode='lines', line=dict(color='gray', width=line_width), name='Non-Irrigation Season Discharge', connectgaps=False, showlegend=True ))
        irrigation_marker_traces = []; non_irrigation_marker_traces = []; can_plot_seasons = True
        irrigation_season_data = df[df['Season'] == 'Irrigation'].copy(); non_irrigation_season_data = df[df['Season'] == 'Non-Irrigation'].copy()
        for flag, (color, legend_name) in flag_colors.items():
            if flag in irrigation_season_data.columns: # Irrigation
                subset = irrigation_season_data[irrigation_season_data[flag].fillna(False).astype(bool)];
                if not subset.empty: irrigation_marker_traces.append(go.Scatter( x=subset['Date'].tolist(), y=subset['DISCHARGE'].tolist(), mode='markers', marker=dict(color=color, size=7), name=legend_name, legendgroup=flag, showlegend=True, visible=True ))
            if flag in non_irrigation_season_data.columns: # Non-Irrigation
                subset = non_irrigation_season_data[non_irrigation_season_data[flag].fillna(False).astype(bool)];
                if not subset.empty: non_irrigation_marker_traces.append(go.Scatter( x=subset['Date'].tolist(), y=subset['DISCHARGE'].tolist(), mode='markers', marker=dict(color=color, size=7), name=legend_name, legendgroup=flag, showlegend=True, visible=False ))
        for trace in irrigation_marker_traces: fig.add_trace(trace)
        for trace in non_irrigation_marker_traces: fig.add_trace(trace)

        # 4. Define Button Logic (unchanged)
        num_irr_markers = len(irrigation_marker_traces); num_nonirr_markers = len(non_irrigation_marker_traces); num_total_traces = 2 + num_irr_markers + num_nonirr_markers
        def create_visibility(show_irr_line, show_nonirr_line, show_irr_markers, show_nonirr_markers): visibility = [False] * num_total_traces; visibility[0] = show_irr_line; visibility[1] = show_nonirr_line; [visibility.__setitem__(i, True) for i in range(2, 2 + num_irr_markers) if show_irr_markers]; [visibility.__setitem__(i, True) for i in range(2 + num_irr_markers, num_total_traces) if show_nonirr_markers]; return visibility
        irrigation_visible_args = create_visibility(True, False, True, False); non_irrigation_visible_args = create_visibility(False, True, False, True); all_seasons_visible_args = create_visibility(True, True, True, True)
        irrigation_button = dict(label="Irrigation", method="update", args=[{"visible": irrigation_visible_args}])
        non_irrigation_button = dict(label="Non-Irrigation", method="update", args=[{"visible": non_irrigation_visible_args}])
        all_seasons_button = dict(label="All Seasons", method="update", args=[{"visible": all_seasons_visible_args}])

        # 5. Calculate Overall Axis Ranges AND Custom Year Ticks/Labels
        x_range = None; y_range = None; x_tickvals = None; x_ticktext = None
        if not df['Date'].empty:
            min_date = df['Date'].min()
            max_date = df['Date'].max()
            x_range = [min_date, max_date] # Keep overall range

            # Calculate specific tick values (Jan 1st) and text (Year)
            min_year = min_date.year
            max_year = max_date.year
            # Ensure reasonable number of ticks; maybe only every few years if span is large
            year_step = 1
            if (max_year - min_year) > 15: # Example threshold: if > 15 years, label every 2 years
                year_step = 2
            if (max_year - min_year) > 30: # Example threshold: if > 30 years, label every 5 years
                 year_step = 5

            years_to_label = range(min_year, max_year + 1, year_step)
            x_tickvals = [pd.Timestamp(f'{year}-01-01') for year in years_to_label]
            x_ticktext = [str(year) for year in years_to_label]

            # Filter ticks to be within or very close to the actual data range
            # This prevents ticks appearing way before/after the plotted data
            x_tickvals_filtered = []
            x_ticktext_filtered = []
            # Add buffer (e.g., half a year) to range for filtering ticks near edges
            range_buffer = pd.Timedelta(days=180)
            filter_min_date = min_date - range_buffer
            filter_max_date = max_date + range_buffer

            for tv, tt in zip(x_tickvals, x_ticktext):
                if filter_min_date <= tv <= filter_max_date:
                    x_tickvals_filtered.append(tv)
                    x_ticktext_filtered.append(tt)
            # Use filtered lists if filtering produced results, else maybe fallback (though unlikely needed)
            if x_tickvals_filtered:
                 x_tickvals = x_tickvals_filtered
                 x_ticktext = x_ticktext_filtered

        if df['DISCHARGE'].notna().any():
            # (y_range calculation remains the same)
            y_min_data = df['DISCHARGE'].min(); y_max_data = df['DISCHARGE'].max()
            final_y_min = 0 if y_min_data >= 0 else y_min_data * 1.05; final_y_max = y_max_data * 1.05 if y_max_data > 0 else (y_max_data * 0.95 if y_max_data < 0 else 1)
            if final_y_min >= final_y_max: final_y_min -= 1; final_y_max += 1
            y_range = [final_y_min, final_y_max]


        # 6. Update Figure Layout <-- Apply custom axis ticks here
        fig.update_layout(
            title=dict(text=plot_title, x=0.5, y=0.98, font=dict(size=20)),
            yaxis=dict(title=f"Mean Daily Discharge ({units})", title_font=dict(size=18), tickfont=dict(size=14), range=y_range),
            xaxis=dict(
                title="Date", title_font=dict(size=18), tickfont=dict(size=14),
                range=x_range,         # Keep overall fixed range
                # ***** START: Custom Tick Configuration *****
                tickmode='array',      # Use explicit tick values/text
                tickvals=x_tickvals,   # Set the calculated tick positions (Jan 1st dates)
                ticktext=x_ticktext    # Set the calculated tick labels (Year strings)
                # ***** END: Custom Tick Configuration *****
            ),
            legend=dict( orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, title=dict(text="Flagging Criteria:", font=dict(size=16)), font=dict(size=12), tracegroupgap=5 ),
            updatemenus=[ dict( type="buttons", direction="left", buttons=[irrigation_button, non_irrigation_button, all_seasons_button], showactive=True, x=0.01, xanchor="left", y=1.05, yanchor="bottom" ) ] if can_plot_seasons and (irrigation_marker_traces or non_irrigation_marker_traces) else [],
            template="plotly_white", margin=dict(t=100, r=250), width=1500, height=800 )

        # 7. Convert to JSON
        plot_json = fig.to_dict()

    except Exception as e:
        tb_str = traceback.format_exc(); print(tb_str)
        error_message = f"An error occurred during data processing or plot generation: {str(e)}"

    # --- Rendering ---
    return render_template_string(HTML_TEMPLATE, plot_json=plot_json, site_id=site_id, error=error_message, warning=warning_message, nodata=nodata_message)

# --- Main execution block ---
if __name__ == '__main__':
    print("Starting Flask development server...")
    print(f"Access the plot via http://localhost:8080/plot?id=YOUR_SITE_ID (e.g., http://localhost:8080/plot?id=10987)")
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
