import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from flask import Flask, request, render_template_string
import plotly.graph_objects as go
import traceback
import json # Added for converting fig to json robustly

app = Flask(__name__)

# (HTML_TEMPLATE remains the same as in the old script)
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
    plot_json_str = None # Will store the JSON string for the template
    error_message = None
    warning_message = None
    nodata_message = None

    if not site_id:
        error_message = "Missing 'id' parameter in query string."
        return render_template_string(HTML_TEMPLATE, error=error_message, site_id="N/A"), 400

    # --- Data Fetching ---
    try:
        end_date = datetime.today().strftime("%Y-%m-%d")
        api_url = f"https://www.waterrights.utah.gov/dvrtdb/daily-chart.asp?station_id={site_id}&end_date={end_date}&f=json"
        print(f"Fetching data for site {site_id}: {api_url}") # Keep for debugging
        response = requests.get(api_url, timeout=30) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # Check if 'data' key exists and is a list *before* trying to access it
        if "data" not in data or not isinstance(data["data"], list):
            error_message = f"API response 'data' field is invalid or missing for site {site_id}. Response: {response.text[:500]}" # Include part of response
            return render_template_string(HTML_TEMPLATE, error=error_message, site_id=site_id), 502

    except requests.exceptions.RequestException as e:
        tb_str = traceback.format_exc(); print(tb_str)
        error_message = f"Error fetching data from API: {str(e)}"
        return render_template_string(HTML_TEMPLATE, error=error_message, site_id=site_id), 500
    except Exception as e: # Catch other potential errors during fetch/initial JSON parse
        tb_str = traceback.format_exc(); print(tb_str)
        error_message = f"An unexpected error occurred during data fetching: {str(e)}"
        return render_template_string(HTML_TEMPLATE, error=error_message, site_id=site_id), 500


    # --- Data Processing & Plotting (using New Script Logic) ---
    try:
        if not data["data"]: # Check if the data list is empty
            nodata_message = f"No time series data returned from API for site {site_id} for the requested period."
            # Return 200 OK, but indicate no data
            return render_template_string(HTML_TEMPLATE, nodata=nodata_message, site_id=site_id), 200
        else:
            # --- Start: Adapted from New Script ---
            df = pd.DataFrame(data["data"], columns=["date", "value"])
            df.rename(columns={"date": "Date", "value": "DISCHARGE"}, inplace=True)

            # Add metadata as new columns BEFORE cleaning/dropping rows
            metadata_fields = ["station_id", "station_name", "system_name", "units"]
            metadata = {field: data.get(field, "N/A") for field in metadata_fields}
            for key, value in metadata.items():
                 df[key] = value

            # Convert types and handle initial NaNs AFTER metadata is added
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['DISCHARGE'] = pd.to_numeric(df['DISCHARGE'], errors='coerce')

            # Drop rows where essential Date or DISCHARGE is NaN AFTER conversion attempts
            # Keep original index temporarily if needed for merging later, but reset for processing
            df = df.dropna(subset=['Date', 'DISCHARGE']).sort_values(by='Date').reset_index(drop=True)

            # Check if DataFrame is empty *after* cleaning
            if df.empty:
                nodata_message = f"No valid, plottable data points found for site {site_id} after cleaning."
                return render_template_string(HTML_TEMPLATE, nodata=nodata_message, site_id=site_id), 200


            # Reorder columns (example, adjust as needed)
            core_cols = ["Date", "DISCHARGE"]
            meta_cols = [col for col in metadata_fields if col in df.columns]
            df = df[core_cols + meta_cols] # Basic reordering


            # === FLAGGING CRITERIA (from New Script) === #
            # Exclude 0 values when flagging for negative values
            df['FLAG_NEGATIVE'] = (df['DISCHARGE'] < 0) & (df['DISCHARGE'] != 0)

            # Exclude 0 values for "FLAG_ZERO" criteria
            df['FLAG_ZERO'] = (df['DISCHARGE'] == 0)

            # === Filter Out Non-Positive Values for Statistical Tests === #
            df_nonzero = df[df["DISCHARGE"] > 0].copy()

            if not df_nonzero.empty:
                # === Compute IQR Thresholds === #
                Q1, Q3 = df_nonzero["DISCHARGE"].quantile([0.25, 0.75])
                IQR = Q3 - Q1

                # === Compute 90th Percentile Thresholds === #
                discharge_90th_percentile = np.percentile(df_nonzero["DISCHARGE"].dropna(), 90)

                # === Compute the 95th Percentile Thresholds === #
                discharge_95th_percentile = np.percentile(df_nonzero["DISCHARGE"].dropna(), 95)

                # === Compute Rate of Change and Merge Back === #
                # Ensure date sorting before diff
                df_nonzero = df_nonzero.sort_values(by="Date")
                df_nonzero["RATE_OF_CHANGE"] = df_nonzero["DISCHARGE"].diff().abs()
                # Merge RoC back to the original df based on index (more robust than date string if dates weren't unique initially)
                df = df.merge(df_nonzero[["RATE_OF_CHANGE"]], left_index=True, right_index=True, how="left")


                # === Flag Repeated Values (4 or more in a row) === #
                # Use the original df for this, considering only non-NaN, non-zero for the *value check* but apply flag to original df index
                # Create groups based on consecutive identical values *within the non-zero subset*
                non_zero_mask = df["DISCHARGE"] > 0
                if non_zero_mask.any():
                     is_different = (df.loc[non_zero_mask, "DISCHARGE"] != df.loc[non_zero_mask, "DISCHARGE"].shift())
                     # Calculate group sizes based on consecutive identical values in the non-zero subset
                     group_sizes = df.loc[non_zero_mask, "DISCHARGE"].groupby(is_different.cumsum()).transform("size")
                     # Create the flag column in the original DataFrame, defaulting to False
                     df["FLAG_REPEATED"] = False
                     # Apply the flag where the group size is >= 4 *on the corresponding rows* in the original df
                     df.loc[non_zero_mask & (group_sizes >= 4), "FLAG_REPEATED"] = True
                else:
                     df["FLAG_REPEATED"] = False # No non-zero values to repeat


                # === Run Isolation Forest for Outlier Detection === #
                # Ensure DISCHARGE has no NaNs before fitting (already handled by df_nonzero creation)
                df_iforest = df_nonzero[["DISCHARGE"]].dropna()
                if not df_iforest.empty and df_iforest['DISCHARGE'].nunique() > 1: # Check for variability
                    model = IsolationForest(contamination=0.05, random_state=42)
                    # Fit on non-NaN values and assign back using index of df_nonzero
                    df_nonzero["OUTLIER_IF_PREDICT"] = model.fit_predict(df_iforest)
                    df_nonzero["OUTLIER_IF"] = df_nonzero["OUTLIER_IF_PREDICT"] == -1
                    # Merge back to original df
                    df = df.merge(df_nonzero[["OUTLIER_IF"]], left_index=True, right_index=True, how="left")
                    df["OUTLIER_IF"] = df["OUTLIER_IF"].fillna(False).astype(bool) # Fill NA and ensure boolean
                else:
                    df["OUTLIER_IF"] = False # Handle case with no data or no variability


                # === Compute Relative Standard Deviation for Flagging Large Spikes === #
                mean_discharge = df_nonzero["DISCHARGE"].mean()
                # Avoid division by zero if mean is zero
                if mean_discharge != 0:
                    # Calculate on original df, using the mean from non-zero
                    df["PERCENT_DEV"] = ((df["DISCHARGE"] - mean_discharge).abs() / mean_discharge) * 100
                else:
                    df["PERCENT_DEV"] = np.nan # Or set to 0 or another appropriate value

                # === Flag Discharges with Large Relative Deviation === #
                threshold = 1000 # Over 1000% deviation
                # Apply flag where calculation was possible and condition met
                df["FLAG_RSD"] = (df["PERCENT_DEV"] > threshold) & (df["DISCHARGE"] != 0) & df["PERCENT_DEV"].notna()
                df["FLAG_RSD"] = df["FLAG_RSD"].fillna(False) # Fill remaining NaNs if any

                # Note: Merging for IF and Repeated flags already done above.

            else:
                # === Fallback Values for Empty or Non-Positive Series === #
                discharge_90th_percentile = np.nan # Use NaN to indicate it wasn't calculated
                discharge_95th_percentile = np.nan
                IQR = np.nan
                Q1 = Q3 = np.nan
                df["RATE_OF_CHANGE"] = np.nan
                df["OUTLIER_IF"] = False
                df["FLAG_REPEATED"] = False
                df["PERCENT_DEV"] = np.nan
                df["FLAG_RSD"] = False

            # === Flag Values > 95th Percentile Outliers, IQR Outliers Upper and Lower Bound, Flag Values Rate of Change > 90th Percentile Value === #
            # Ensure thresholds are numeric (not NaN) before comparison
            if pd.notna(discharge_95th_percentile):
                 # Apply only to non-zero values
                df["FLAG_Discharge"] = (df["DISCHARGE"] > discharge_95th_percentile) & (df["DISCHARGE"] > 0)
            else:
                df["FLAG_Discharge"] = False

            # Ensure IQR components are numeric (not NaN)
            if pd.notna(Q1) and pd.notna(Q3) and pd.notna(IQR) and IQR > 0: # Check IQR > 0 to avoid issues
                 # Apply only to non-zero values
                df["FLAG_IQR"] = ((df["DISCHARGE"] < Q1 - 1.5 * IQR) | (df["DISCHARGE"] > Q3 + 1.5 * IQR)) & (df["DISCHARGE"] > 0)
            elif pd.notna(Q1) and IQR == 0: # Handle zero IQR case (all non-zero values are the same)
                df["FLAG_IQR"] = (df["DISCHARGE"] != Q1) & (df["DISCHARGE"] > 0)
            else: # Handles NaN IQR or other issues
                df["FLAG_IQR"] = False


            # Ensure RoC threshold is numeric (not NaN)
            if pd.notna(discharge_90th_percentile):
                # Apply where RoC is calculated and > threshold
                df["FLAG_RoC"] = (df["RATE_OF_CHANGE"] > discharge_90th_percentile) & df["RATE_OF_CHANGE"].notna()
            else:
                df["FLAG_RoC"] = False


            # Fill NaNs potentially introduced by merges or calculations in flag columns before combining
            flag_cols_to_fill = ["FLAG_NEGATIVE", "FLAG_ZERO", "FLAG_REPEATED", "OUTLIER_IF",
                                 "FLAG_RSD", "FLAG_Discharge", "FLAG_IQR", "FLAG_RoC"]
            for col in flag_cols_to_fill:
                 if col in df.columns:
                     df[col] = df[col].fillna(False).astype(bool) # Ensure boolean after filling
                 else:
                     df[col] = False # Add column if it wasn't created due to edge cases

            # === Above Max (Suspected Above Max)--only show common flagged values among the three methods (IQR, > 95th Percentile, IF) === #
            df["FLAG_ABOVE_MAX_OVERLAP"] = (
                df["FLAG_IQR"] &
                df["FLAG_Discharge"] &
                df["OUTLIER_IF"]
            ) & (df["DISCHARGE"] > 0) # Ensure we only flag positive values here

            # === Large Spikes --only show common flagged values among the two methods (RSD,RoC) === #
            df["FLAG_LARGE_SPIKES"] = df["FLAG_RSD"] & df["FLAG_RoC"]


            # === Overall Flagged Record Indicator === #
            # Define the specific flags to check for the final "FLAGGED" status
            flags_for_overall = ["FLAG_NEGATIVE", "FLAG_ZERO", "FLAG_REPEATED",
                                 "FLAG_ABOVE_MAX_OVERLAP", "FLAG_LARGE_SPIKES"]
            df["FLAGGED"] = df[flags_for_overall].any(axis=1)


            # --- Plotting Setup (from New Script) ---
            plot_title = f"Flagged Data Points for {metadata.get('station_name', 'Station ' + site_id)}"
            units = metadata.get('units', 'CFS') # Get units for y-axis label

            # Define custom legend names and colors
            flag_colors = {
                'FLAG_NEGATIVE': ('red', 'Negative (-)'),
                'FLAG_ZERO': ('blue', 'Value = 0'),
                'FLAG_REPEATED': ('green', 'Repeated (≥4 days)'),
                'FLAG_ABOVE_MAX_OVERLAP': ('brown', 'Above Suspected Max'),
                'FLAG_LARGE_SPIKES': ('orange', 'Large Spikes'),
            }

            # Create plot
            fig = go.Figure()

            # Background Line (Mean Daily Discharge)
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df['DISCHARGE'],
                mode='lines',
                line=dict(color='lightgray', width=1.5),
                name='Mean Daily Discharge',
                connectgaps=False # Do not connect across NaN gaps (already dropped NaNs earlier, but good practice)
            ))

            # Add flagged points
            any_flags_plotted = False
            for flag, (color, legend_name) in flag_colors.items():
                 # Explicitly cast to bool after fillna
                 subset = df[df[flag].fillna(False).astype(bool)]
                 if not subset.empty:
                     any_flags_plotted = True
                     fig.add_trace(go.Scatter(
                         x=subset['Date'], y=subset['DISCHARGE'],
                         mode='markers',
                         marker=dict(color=color, size=7),
                         name=legend_name
                     ))

            # Add horizontal dashed line for min value of Above Max
            # Ensure flag column exists before filtering
            if "FLAG_ABOVE_MAX_OVERLAP" in df.columns:
                above_max_subset = df[df["FLAG_ABOVE_MAX_OVERLAP"].fillna(False).astype(bool)]
                if not above_max_subset.empty:
                    min_above_max = above_max_subset["DISCHARGE"].min()
                    # Check if min_above_max is finite before plotting
                    if pd.notna(min_above_max) and np.isfinite(min_above_max):
                        fig.add_trace(go.Scatter(
                            x=[df["Date"].min(), df["Date"].max()],
                            y=[min_above_max, min_above_max],
                            mode="lines",
                            line=dict(color="brown", width=2, dash="dash"),
                            name="Min Value (Above Suspected Max)"
                        ))


            # Update layout with title and legend placement (from New Script)
            fig.update_layout(
                title=dict(
                    text=plot_title,
                    x=0.5,  # Centers the title
                    font=dict(size=20)
                ),
                xaxis=dict(
                    title="Date",
                    title_font=dict(size=18),
                    tickfont=dict(size=14)
                    # Removed custom tick logic from old script for simplicity
                ),
                yaxis=dict(
                    title=f"Mean Daily Discharge ({units})", # Use units from metadata
                    title_font=dict(size=18),
                    tickfont=dict(size=14)
                    # Removed fixed y-range from old script, let Plotly auto-scale
                ),
                legend=dict(
                    orientation="h",  # Horizontal legend
                    yanchor="top",
                    y=-0.15,  # Adjusted position slightly below plot
                    xanchor="center",
                    x=0.5,
                    title=dict(text="Flagging Criteria:", font=dict(size=16)),
                    font=dict(size=14)
                ),
                template="plotly_white",
                width=1400, # Adjust width as needed
                height=700, # Adjust height as needed
                margin=dict(t=80, b=120) # Adjust margins (bottom increased for legend)
            )

            # Add warning if no flags were triggered to plot
            if not any_flags_plotted and not nodata_message:
                warning_message = f"Data processed successfully for site {site_id}, but no data points met the flagging criteria."

            # --- End: Adapted from New Script ---

            # Convert figure to JSON string for embedding in HTML
            # Use plotly's json encoder for robustness with numpy types etc.
            plot_json_str = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


    except Exception as e:
        # Catch errors during the processing/plotting phase
        tb_str = traceback.format_exc(); print(tb_str)
        error_message = f"An error occurred during data processing or plot generation: {str(e)}"
        # Return 500 Internal Server Error
        return render_template_string(HTML_TEMPLATE, error=error_message, site_id=site_id), 500


    # --- Rendering ---
    # Pass the JSON string to the template
    return render_template_string(HTML_TEMPLATE, plot_json_str=plot_json_str, site_id=site_id, error=error_message, warning=warning_message, nodata=nodata_message)


# --- Main execution block ---
if __name__ == '__main__':
    print("Starting Flask development server...")
    # Use 0.0.0.0 to make it accessible on the network, default is 127.0.0.1
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8080))
    print(f"Access the plot via http://{host}:{port}/plot?id=YOUR_SITE_ID (e.g., http://localhost:{port}/plot?id=10987)")
    # Set debug=False for production, True for development (enables auto-reload)
    debug_mode = os.environ.get("FLASK_DEBUG", "True").lower() == "true"
    app.run(debug=debug_mode, host=host, port=port)
