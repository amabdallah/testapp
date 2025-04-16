# --- Imports ---
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from flask import Flask, render_template_string, request
import traceback
import logging
import json # Used for safe JS data embedding if needed

# --- Pandas Option ---
pd.set_option('future.no_silent_downcasting', True) # Address FutureWarning

# --- Flask App Setup ---
app = Flask(__name__)
# Configure logging
logging.basicConfig(level=logging.INFO) # Use INFO level for general flow, DEBUG for more detail

# --- Helper Function for Date Validation ---
def validate_date(date_str):
    """Attempts to parse a date string (YYYY-MM-DD) and returns a datetime object or None."""
    if not date_str:
        return None
    try:
        # Ensure format includes century for robustness
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return None

# --- HTML Template ---
# **** MODIFIED: REMOVED SCRIPT TAG FOR PLOTLY LIBRARY ****
# **** JS Adjusted to find plot div inside container ****
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{ station_name | default('Data Quality Analysis', true) }} (Site ID {{ site_id | default('N/A', true) }}) {{ start_date }} to {{ end_date }}</title>
    <style>
        /* --- CSS --- */
        body { font-family: sans-serif; margin: 20px; }
        h1 { text-align: center; margin-block-start: 0.67em; margin-block-end: 0.67em; line-height: 1.3; }
        .error { color: red; text-align: center; font-weight: bold; margin-top: 15px; }
        /* Using plot_container ID for styling if needed */
        #plot_container { margin-top: 20px; min-height: 100px; background-color: #f0f0f0; }
        .controls { text-align: center; margin-bottom: 20px; padding: 15px; border: 1px solid #ccc; border-radius: 5px; background-color: #f9f9f9; }
        .controls label, .controls input, .controls button { margin: 0 5px; vertical-align: middle; }
        .controls input[type="submit"], .controls button { padding: 5px 15px; cursor: pointer; font-size: 1em; }
        .plot-title-info { text-align: center; font-size: 16px; margin-bottom: 10px; }
        .header-link { font-size: 14px; color: darkblue; text-decoration: none; }
        .header-link:hover { text-decoration: underline; }
        .header-text-no-link { font-size: 14px; color: darkblue; }
        .modal { display: none; position: fixed; z-index: 1000; left: 50%; top: 50%; transform: translate(-50%, -50%); width: 300px; max-width: 90%; padding: 20px; background-color: #fefefe; border: 3px solid red; /* Debug border */ box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); border-radius: 5px; text-align: center; }
        .modal-content { position: relative; }
        .modal-content h4 { margin-top: 0; }
        .modal-content p { margin: 10px 0; font-size: 14px; word-wrap: break-word; }
        .modal-close { color: #aaa; position: absolute; top: 5px; right: 10px; font-size: 24px; font-weight: bold; line-height: 1; cursor: pointer; padding: 0 5px; }
        .modal-close:hover, .modal-close:focus { color: black; text-decoration: none; }
        .modal-button { display: inline-block; padding: 5px 10px; font-size: 12px; font-weight: bold; font-family: sans-serif; margin: 5px 3px; cursor: pointer; background-color: #e7e7e7; color: #333; border: 1px solid #adadad; border-radius: 4px; text-decoration: none; text-align: center; line-height: 1.4; white-space: nowrap; box-shadow: 0 1px 1px rgba(0,0,0,0.1); -webkit-appearance: button; -moz-appearance: button; appearance: button; }
        .modal-button:hover { background-color: #dcdcdc; border-color: #999999; text-decoration: none; color: #000; box-shadow: 0 1px 1px rgba(0,0,0,0.2); }
        .modal-button:active { background-color: #cccccc; box-shadow: inset 0 1px 2px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <h1>
        <span style="font-size: 18px;">Data Quality Analysis for Measurement Site</span><br>
        {% if site_id and site_id != 'N/A' and site_id is not none %}
            <a href="https://waterrights.utah.gov/cgi-bin/dvrtview.exe?Modinfo=StationView&STATION_ID={{ site_id }}" target="_blank" rel="noopener noreferrer" class="header-link">
                {{ station_name | default('Unknown Station', true) }} (Site ID={{ site_id }})
            </a>
        {% else %}
            <span class="header-text-no-link">
                {{ station_name | default('Unknown Station', true) }} (Site ID={{ site_id | default('N/A', true) }})
            </span>
        {% endif %}
    </h1>
    <div class="controls">
        <form method="GET" action="/plot">
            <label for="id">Site ID:</label>
            <input type="text" id="id" name="id" value="{{ site_id | default('', true) }}" required>
            <label for="start_date">Start Date:</label>
            <input type="date" id="start_date" name="start_date" value="{{ start_date | default('', true) }}" required>
            <label for="end_date">End Date:</label>
            <input type="date" id="end_date" name="end_date" value="{{ end_date | default('', true) }}" required>
            <input type="submit" value="Update Plot">
            <button type="button" onclick="resetDates()">Reset Dates</button>
        </form>
    </div>
    {% if error %} <p class="error">Error: {{ error }}</p> {% endif %}


    {# --- Plot and Modal Section --- #}
    {% if plot_div %}
        <div id='plot_container'>
             {{ plot_div | safe }}
        </div>
        <script>console.log("[HTML] Plot div content should be inside plot_container above this line.");</script>

        <div id="pointActionModal" class="modal">
             <div class="modal-content">
                 <span id="closeModal" class="modal-close" title="Close">&times;</span>
                 <h4>Point Action</h4>
                 <p id="modalPointInfo">Point details will appear here.</p>
                 <div id="modalActions"></div>
             </div>
        </div>

        <script>
            // --- Util Functions ---
             function resetDates() {
                var siteIdInput = document.getElementById('id');
                if (!siteIdInput) { console.error("Site ID input field not found."); alert("Internal error: Cannot find Site ID field."); return; }
                var siteId = siteIdInput.value;
                if (!siteId || siteId.trim() === "") { alert("Please enter a Site ID first before resetting dates."); return; }
                window.location.href = '/plot?id=' + encodeURIComponent(siteId) + '&reset=true';
            }

            function pointAction(action, pointIndex, xValue, yValue) {
                 console.log("Action:", action, "Index:", pointIndex, "Date:", xValue, "Value:", yValue);
                 let displayValue = typeof yValue === 'number' ? yValue.toFixed(2) : yValue;
                 alert(action + " clicked for point:\\nDate: " + xValue + "\\nValue: " + displayValue + "\\n(Point Index: " + pointIndex + ")\\n\\nAction not yet implemented.");
                 var modal = document.getElementById('pointActionModal');
                 if (modal) { modal.style.display = "none"; }
            }

            // Flag to prevent infinite retries (kept just in case, but less likely needed now)
            window.plotInteractionRetry = false;

            function initializePlotInteraction() {
                console.log("[Init] Starting interaction setup (include_plotlyjs='cdn' mode)...");

                // The plot div ID is often dynamic when including JS. Find it reliably.
                var plotContainer = document.getElementById('plot_container');
                var plotDiv = null; // Initialize plotDiv

                if (!plotContainer) {
                     console.error("[Init] Plot container (#plot_container) not found! Cannot find plot div.");
                     return; // Cannot proceed
                }

                // Attempt to find the plot div using common selectors Plotly uses
                // Wait slightly for Plotly's embedded JS to potentially create the div
                setTimeout(function() {
                    console.log("[Init] Attempting to find plot div inside #plot_container after short delay...");
                    plotDiv = plotContainer.querySelector('div.js-plotly-plot'); // Common class
                    if (!plotDiv) {
                        plotDiv = plotContainer.querySelector('div.plotly-graph-div'); // Alternative class
                    }
                    if (!plotDiv) {
                        // Fallback: Grab the first div child of the container
                        var childDivs = plotContainer.getElementsByTagName('div');
                        if (childDivs.length > 0) {
                           plotDiv = childDivs[0];
                           console.warn("[Init] Couldn't find plot div by class, using first child div:", plotDiv);
                        }
                    }

                    if (!plotDiv) {
                        console.error("[Init] Plot div element NOT FOUND within #plot_container even after delay. Cannot attach listener.");
                        // Check if Plotly object exists - indicates library IS loaded but failed to render/find div
                        if(typeof Plotly !== 'undefined') {
                             console.warn("[Init] Plotly library IS loaded, but the plot div element wasn't found by selectors.");
                        } else {
                             // This case should be rare now as Plotly is included in the div
                             console.error("[Init] Plotly library is ALSO not loaded (unexpected in this mode).");
                        }
                        return; // Stop if plot div not found
                    } else {
                        console.log("[Init] Found plotDiv element:", plotDiv);
                        // Now that we found plotDiv, proceed with attaching listeners
                        attachPlotlyListeners(plotDiv);
                    }
                }, 100); // Delay (e.g., 100ms) to allow embedded JS to potentially render

            } // End of initializePlotInteraction

            function attachPlotlyListeners(plotDiv) {
                 console.log("[Attach] Attempting to attach listeners to:", plotDiv);

                 var modal = document.getElementById('pointActionModal');
                 var closeModalBtn = document.getElementById('closeModal');
                 var modalPointInfo = document.getElementById('modalPointInfo');
                 var modalActionsDiv = document.getElementById('modalActions');

                 // Attach Modal Close Handlers (should happen only once)
                 if (!modal || !closeModalBtn || !modalPointInfo || !modalActionsDiv) {
                    console.error("[Attach] Modal elements missing.");
                 } else if (!window.modalHandlersAttached) { // Check flag
                    closeModalBtn.onclick = function() { if(modal) modal.style.display = "none"; };
                    window.addEventListener('click', function(event) {
                         if (modal && modal.style.display === "block" && event.target === modal) {
                             if (!plotDiv.contains(event.target) || event.target === plotDiv ) {
                                  modal.style.display = "none";
                             }
                         }
                    });
                    console.log("[Attach] Modal close handlers attached.");
                    window.modalHandlersAttached = true; // Set flag
                 }


                 // --- Attach Plotly Event Listener ---
                 try {
                     // Check if plotDiv has '.on' method. It should if Plotly initialized it.
                     if (typeof plotDiv.on !== 'function') {
                         throw new Error("plotDiv.on is still not a function. Plotly failed to initialize this div correctly.");
                     }

                     console.log("[Attach] Attaching 'plotly_click' listener...");
                     plotDiv.on('plotly_click', function(data) {
                         console.log("==== Plotly CLICK Event Fired ====");
                         console.log("[plotly_click] Received data:", data);

                         if (!modal || !modalPointInfo || !modalActionsDiv) {
                             console.error("[plotly_click] Modal elements missing inside handler."); return;
                         }
                         if (!data || !data.points || data.points.length === 0) {
                             console.log("[plotly_click] Click was not on a data point."); return;
                         }

                         var pointData = data.points[0];
                         if (pointData.curveNumber > 0) {
                             console.log("[plotly_click] Clicked on a flagged point (curveNumber > 0).");
                             var pIndex = pointData.pointNumber;
                             var pX = pointData.x; var pY = pointData.y;
                             var pTraceName = pointData.fullData ? pointData.fullData.name : 'Unknown Trace';
                             var displayX = pX;
                             var displayY = typeof pY === 'number' ? pY.toFixed(2) : String(pY);
                             var flagType = String(pTraceName).split('[')[0].trim();

                             modalPointInfo.innerHTML = `<b>Date:</b> ${displayX}<br><b>Value:</b> ${displayY}<br><b>Flag:</b> ${flagType}`;
                             modalActionsDiv.innerHTML = ''; // Clear
                             var btnIgnore = document.createElement('button'); btnIgnore.className = 'modal-button'; btnIgnore.innerText = 'Ignore'; btnIgnore.onclick = () => pointAction('Ignore', pIndex, pX, pY);
                             var btnInterpolate = document.createElement('button'); btnInterpolate.className = 'modal-button'; btnInterpolate.innerText = 'Interpolate'; btnInterpolate.onclick = () => pointAction('Interpolate', pIndex, pX, pY);
                             modalActionsDiv.appendChild(btnIgnore); modalActionsDiv.appendChild(btnInterpolate);

                             modal.style.display = 'block';
                             console.log("[plotly_click] Modal displayed with point info.");
                         } else {
                             console.log("[plotly_click] Clicked on the base line (curveNumber 0).");
                         }
                     });
                     console.log("[Attach] 'plotly_click' listener attached successfully.");

                     plotDiv.on('plotly_afterplot', function() {
                          console.log("---- Plotly AFTERPLOT Event Fired ----");
                     });
                     console.log("[Attach] 'plotly_afterplot' listener attached successfully.");

                 } catch(err) {
                     console.error("[Attach] FAILED to attach listener:", err);
                     console.log("[Attach] PlotDiv object during failure:", plotDiv);
                     alert("Error attaching plot listener. Check console.");
                 }
            } // End of attachPlotlyListeners


            // --- Run Initialization ---
            // Use DOMContentLoaded to ensure #plot_container and modal elements exist
             if (document.readyState === 'loading') {
                 console.log("DOM not ready yet, adding listener for DOMContentLoaded.");
                document.addEventListener('DOMContentLoaded', initializePlotInteraction);
            } else {
                 console.log("DOM already ready, running initializePlotInteraction directly.");
                initializePlotInteraction();
            }
        </script>
        {# --- Fallback messages etc --- #}
    {% elif not error and not site_id %}
        <p style="text-align: center;">Please enter a Site ID and select a date range above.</p>
    {% elif not error and site_id and not plot_div %}
        <p style="text-align: center;">No plot generated. Check if data exists for the selected Site ID and date range, or if there was an error fetching/processing data.</p>
    {% endif %}

</body>
</html>
"""
# **** END OF HTML TEMPLATE VARIABLE ****


# --- Core Data Processing and Plotting Function ---
# (Ensure this function is exactly as provided in the previous responses
#  where it correctly processed data and created the Plotly figure object)
def generate_plot_for_site(site_id, start_date_str_requested, end_date_str_requested, is_reset=False):
    station_name = None
    actual_start_date_str = start_date_str_requested
    actual_end_date_str = end_date_str_requested
    df = pd.DataFrame()
    metadata = {}

    app.logger.info(f"Initial request for Site ID: {site_id}, Start: {start_date_str_requested}, End: {end_date_str_requested}, Reset: {is_reset}")

    api_end_date_call = datetime.now().strftime('%Y-%m-%d')
    if not is_reset and validate_date(end_date_str_requested):
         api_end_date_call = end_date_str_requested
    elif not is_reset:
         app.logger.warning(f"Invalid/missing end date ('{end_date_str_requested}'). Using today '{api_end_date_call}'.")

    try:
        api_start_date_call = "1900-01-01"
        if not is_reset and validate_date(start_date_str_requested):
            api_start_date_call = start_date_str_requested
        elif not is_reset:
            app.logger.warning(f"Invalid/missing start date ('{start_date_str_requested}'). Using default '{api_start_date_call}'.")

        api_url = f"https://www.waterrights.utah.gov/dvrtdb/daily-chart.asp?station_id={site_id}&start_date={api_start_date_call}&end_date={api_end_date_call}&f=json"
        app.logger.info(f"Fetching data from: {api_url}")
        response = requests.get(api_url, timeout=30)

        if response.status_code != 200:
            err_msg = f"API Error (Status {response.status_code}) for site {site_id}"
            app.logger.error(err_msg + f" URL: {api_url}")
            return None, err_msg, None, start_date_str_requested, end_date_str_requested

        try:
            data = response.json()
            if not isinstance(data, dict): raise ValueError("API response not JSON object.")
        except (requests.exceptions.JSONDecodeError, ValueError) as json_err:
            snippet = response.text[:200] if hasattr(response, 'text') else '(No text)'
            err_msg = f"JSON Decode Error for site {site_id}. Error: {json_err}. Snippet: {snippet}..."
            app.logger.error(err_msg + f" URL: {api_url}")
            return None, err_msg, None, start_date_str_requested, end_date_str_requested

        metadata_fields = ["station_id", "station_name", "system_name", "units"]
        metadata = {f: data.get(f, "N/A") for f in metadata_fields}
        station_name = metadata.get('station_name', 'N/A')
        units = metadata.get('units', 'CFS')

        if "data" not in data or not isinstance(data["data"], list) or not data["data"]:
            err_msg = f"No 'data' array in API response for site {site_id}. Station: {station_name}"
            app.logger.warning(err_msg + f" URL: {api_url}")
            return None, err_msg, station_name, start_date_str_requested, end_date_str_requested

        try:
            df = pd.DataFrame(data["data"], columns=["date", "value"])
        except Exception as df_err:
            err_msg = f"DataFrame creation error for site {site_id}. Error: {df_err}"
            app.logger.error(err_msg, exc_info=True)
            return None, err_msg, station_name, start_date_str_requested, end_date_str_requested

        if df.empty:
            err_msg = f"DataFrame created but empty for site {site_id}. Station: {station_name}"
            app.logger.warning(err_msg + f" URL: {api_url}")
            return None, err_msg, station_name, start_date_str_requested, end_date_str_requested

        if "date" in df.columns and "value" in df.columns:
             df.rename(columns={"date": "Date", "value": "DISCHARGE"}, inplace=True)
        else:
             err_msg = f"Missing 'date' or 'value' column for site {site_id}."
             app.logger.error(err_msg + f" Columns: {df.columns.tolist()}")
             return None, err_msg, station_name, start_date_str_requested, end_date_str_requested

        for k, v in metadata.items():
            if k not in df.columns and k not in ["station_name", "units"]: df[k] = v

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df = df.sort_values('Date').reset_index(drop=True)

        if df.empty:
            err_msg = f"No valid dates after conversion for site {site_id}."
            app.logger.warning(err_msg)
            return None, err_msg, station_name, start_date_str_requested, end_date_str_requested

        min_data_dt = df['Date'].min(); max_data_dt = df['Date'].max()
        app.logger.info(f"Data range: {min_data_dt:%Y-%m-%d} to {max_data_dt:%Y-%m-%d}")

        if is_reset:
            start_dt, end_dt = min_data_dt, max_data_dt
            app.logger.info("Reset: Using full data range.")
        else:
            start_req = validate_date(start_date_str_requested)
            end_req = validate_date(end_date_str_requested)
            if not start_req or not end_req or start_req > end_req:
                app.logger.warning(f"Invalid requested dates. Using full data range.")
                start_dt, end_dt = min_data_dt, max_data_dt
            else:
                start_dt = max(start_req, min_data_dt)
                end_dt = min(end_req, max_data_dt)
                if start_dt > end_dt:
                    app.logger.warning("Requested range outside data bounds. Using full range.")
                    start_dt, end_dt = min_data_dt, max_data_dt

        actual_start_date_str = start_dt.strftime('%Y-%m-%d')
        actual_end_date_str = end_dt.strftime('%Y-%m-%d')

        app.logger.info(f"Filtering between {actual_start_date_str} and {actual_end_date_str}")
        df = df.loc[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)].copy().reset_index(drop=True)

        if df.empty:
            err_msg = f"No data after filtering for site {site_id} [{actual_start_date_str} to {actual_end_date_str}]."
            app.logger.warning(err_msg)
            return None, err_msg, station_name, actual_start_date_str, actual_end_date_str

        df['DISCHARGE'] = pd.to_numeric(df['DISCHARGE'], errors='coerce')
        app.logger.info(f"Processing {len(df)} data points.")

        # === FLAGGING CRITERIA (Same logic as before) ===
        flag_cols = ["FLAG_NEGATIVE", "FLAG_ZERO", "FLAG_REPEATED", "OUTLIER_IF", "FLAG_RSD", "FLAG_Discharge", "FLAG_IQR", "FLAG_RoC", "FLAG_ABOVE_MAX_OVERLAP", "FLAG_LARGE_SPIKES", "FLAGGED"]
        for col in flag_cols: df[col] = False
        df["RATE_OF_CHANGE"] = df['DISCHARGE'].diff().abs()
        df["PERCENT_DEV"] = np.nan
        df['FLAG_NEGATIVE'] = (df['DISCHARGE'].notna()) & (df['DISCHARGE'] < 0)
        df['FLAG_ZERO'] = (df['DISCHARGE'].notna()) & (df['DISCHARGE'] == 0)
        df_nz = df[df['DISCHARGE'].notna() & (df['DISCHARGE'] > 0)].copy()
        df_calc = df_nz.dropna(subset=['DISCHARGE'])
        q90=0; q95=0; Q1=0; Q3=0; IQR=0; mean_val=0
        if not df_calc.empty:
            try:
                Q1, Q3 = df_calc["DISCHARGE"].quantile([0.25, 0.75]).fillna(0)
                IQR = Q3 - Q1
                q90 = df_calc["DISCHARGE"].quantile(0.90).fillna(0)
                q95 = df_calc["DISCHARGE"].quantile(0.95).fillna(0)
                mean_val = df_calc["DISCHARGE"].mean().fillna(0)
                if mean_val != 0: df_nz['PERCENT_DEV'] = ((df_nz["DISCHARGE"] - mean_val).abs() / mean_val * 100)
                df = df.merge(df_nz[['PERCENT_DEV']], left_index=True, right_index=True, how='left')
            except Exception as e: app.logger.warning(f"Stat calc error: {e}")
        if len(df) >= 4:
             is_diff = (df["DISCHARGE"] != df["DISCHARGE"].shift()) | df["DISCHARGE"].isna() | df["DISCHARGE"].shift().isna()
             grp_ids = is_diff.cumsum(); grp_counts = df.groupby(grp_ids)['DISCHARGE'].transform('count')
             df['FLAG_REPEATED'] = (grp_counts >= 4) & df["DISCHARGE"].notna() & (df["DISCHARGE"] != 0)
        else: df['FLAG_REPEATED'] = False
        df_if = df_calc[["DISCHARGE"]].copy()
        if not df_if.empty:
             cont = 'auto' if len(df_if) >= 20 else max(0.001, min(0.5, 0.01))
             model = IsolationForest(contamination=cont, random_state=42)
             try:
                 preds = model.fit_predict(df_if)
                 df_calc['OUTLIER_IF_NZ'] = (preds == -1)
                 df = df.merge(df_calc[['OUTLIER_IF_NZ']], left_index=True, right_index=True, how='left')
                 df['OUTLIER_IF'] = df['OUTLIER_IF_NZ'].fillna(False).astype(bool)
             except ValueError as ife: app.logger.warning(f"IForest error: {ife}"); df['OUTLIER_IF'] = False
        else: df['OUTLIER_IF'] = False
        for col in ["FLAG_Discharge", "FLAG_IQR", "FLAG_RoC", "FLAG_RSD", "OUTLIER_IF", "FLAG_REPEATED"]:
             if col not in df: df[col] = False
             else: df[col] = df[col].fillna(False).astype(bool)
        if q95 > 0: df["FLAG_Discharge"] = (df["DISCHARGE"].notna()) & (df["DISCHARGE"] > q95)
        if IQR >= 0:
             if IQR == 0 and Q1 != 0: df["FLAG_IQR"] = (df["DISCHARGE"].notna()) & (df["DISCHARGE"] != Q1)
             elif IQR > 0: low = Q1 - 1.5 * IQR; high = Q3 + 1.5 * IQR; df["FLAG_IQR"] = (df["DISCHARGE"].notna()) & ((df["DISCHARGE"] < low) | (df["DISCHARGE"] > high))
        if q90 >= 0: df["FLAG_RoC"] = (df["RATE_OF_CHANGE"].notna()) & (df["RATE_OF_CHANGE"] > q90)
        if mean_val != 0: df["FLAG_RSD"] = (df["PERCENT_DEV"].notna()) & (df["PERCENT_DEV"] > 1000)
        df["FLAG_ABOVE_MAX_OVERLAP"] = (df["FLAG_IQR"] & df["FLAG_Discharge"] & df["OUTLIER_IF"])
        df["FLAG_LARGE_SPIKES"] = df["FLAG_RSD"] & df["FLAG_RoC"]
        primary = ["FLAG_NEGATIVE", "FLAG_ZERO", "FLAG_REPEATED", "FLAG_ABOVE_MAX_OVERLAP", "FLAG_LARGE_SPIKES"]
        for col in primary: df[col] = df[col].fillna(False).astype(bool)
        df["FLAGGED"] = df[primary].any(axis=1)
        # === END FLAGGING ===

        # --- Plotting (Same logic as before) ---
        plot_title = f"Data from {actual_start_date_str} to {actual_end_date_str}"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['DISCHARGE'], mode='lines', line=dict(color='lightgray', width=1.5), name='Mean Daily Discharge', connectgaps=False, hoverinfo='skip'))
        hover_tmpl = f'<b>Date:</b> %{{x|%Y-%m-%d}}<br><b>Discharge:</b> %{{y:.2f}} {units}<br><b>Flag Type:</b> %{{fullData.name}}<extra></extra>'
        flags_cfg = {'FLAG_NEGATIVE': ('red', 'Below Capacity (-)'), 'FLAG_ZERO': ('blue', 'Value = 0'), 'FLAG_REPEATED': ('green', 'Repeated (≥4 non-0)'), 'FLAG_ABOVE_MAX_OVERLAP': ('purple', 'Over Capacity'), 'FLAG_LARGE_SPIKES': ('orange', 'Large Spikes')}
        for flag, (color, legend) in flags_cfg.items():
             if flag in df.columns and df[flag].any():
                 subset = df.loc[df[flag]]
                 fig.add_trace(go.Scatter(x=subset['Date'], y=subset['DISCHARGE'], mode='markers', marker=dict(color=color, size=7, symbol='circle'), name=f"{legend} [{len(subset)}]", hovertemplate=hover_tmpl))
        max_mask = df['FLAG_ABOVE_MAX_OVERLAP']
        if max_mask.any():
            min_over = pd.to_numeric(df.loc[max_mask, "DISCHARGE"], errors='coerce').min()
            if pd.notna(min_over) and np.isfinite(min_over):
                min_dt, max_dt = df["Date"].min(), df["Date"].max()
                if pd.notna(min_dt) and pd.notna(max_dt): fig.add_trace(go.Scatter(x=[min_dt, max_dt], y=[min_over, min_over], mode="lines", line=dict(color="purple", width=2, dash="dash"), name="Est. Max Capacity", hoverinfo='skip'))
        fig.update_layout(title=dict(text=plot_title, x=0.5, y=0.95, font_size=16), xaxis_title="Date", yaxis_title=f"Mean Daily Discharge ({units})", legend_title="Suspected Issues", legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"), template="plotly_white", margin=dict(t=80, r=250, b=80, l=80), height=700, hovermode='closest')
        app.logger.info(f"Plot generated: {site_id} [{actual_start_date_str} to {actual_end_date_str}]")
        return fig, None, station_name, actual_start_date_str, actual_end_date_str

    except requests.exceptions.RequestException as e:
        err = f"Network error: {e}"
        app.logger.error(f"API Request failed: {site_id}: {e}")
        name = metadata.get('station_name', 'N/A') if 'metadata' in locals() else None
        return None, err, name, start_date_str_requested, end_date_str_requested
    except Exception as e:
        err = f"Unexpected plot generation error."
        name = metadata.get('station_name', 'N/A') if 'metadata' in locals() else None
        app.logger.error(f"Plot generation error: {site_id}: {e}", exc_info=True)
        return None, err, name, start_date_str_requested, end_date_str_requested


# --- Flask Route: /plot ---
# **** MODIFIED to use include_plotlyjs='cdn' ****
@app.route('/plot')
def show_plot():
    """ Flask route to display the data plot for a given site ID and date range. """
    site_id = request.args.get('id')
    start_date_req = request.args.get('start_date')
    end_date_req = request.args.get('end_date')
    is_reset = request.args.get('reset') == 'true'

    app.logger.info(f"Request: id={site_id}, start={start_date_req}, end={end_date_req}, reset={is_reset}")

    err_msg = None; fig = None; plot_div = None
    st_name = "Data Quality Analysis"; status = 200
    today = datetime.now(); one_yr_ago = today - timedelta(days=365)
    def_start = one_yr_ago.strftime('%Y-%m-%d'); def_end = today.strftime('%Y-%m-%d')
    start_render = start_date_req; end_render = end_date_req

    if not site_id:
        app.logger.info("No Site ID provided.")
        return render_template_string(HTML_TEMPLATE, site_id=None, station_name=st_name, start_date=def_start, end_date=def_end, error=None, plot_div=None)

    if not site_id or not site_id.strip():
         err_msg = "Site ID cannot be empty."; status = 400
         app.logger.warning("Empty Site ID.")
         start_render = start_date_req if start_date_req else def_start
         end_render = end_date_req if end_date_req else def_end
    else:
        start_proc = start_date_req; end_proc = end_date_req
        if not is_reset:
            start_dt = validate_date(start_date_req); end_dt = validate_date(end_date_req)
            if not start_dt or not end_dt:
                 err_msg = "Start/End Dates required (YYYY-MM-DD)."; status = 400
                 app.logger.warning(f"Invalid dates: S='{start_date_req}', E='{end_date_req}'")
                 start_render = start_date_req or ""; end_render = end_date_req or ""
            elif start_dt > end_dt:
                err_msg = "Start date cannot be after end date."; status = 400
                app.logger.warning(f"Date range error: S={start_dt}, E={end_dt}")
                start_render = start_date_req; end_render = end_date_req
            else:
                 start_render = start_dt.strftime('%Y-%m-%d'); end_render = end_dt.strftime('%Y-%m-%d')
                 start_proc = start_render; end_proc = end_render
        else:
            start_proc = None; end_proc = None

        if not err_msg:
            app.logger.info(f"Calling generate_plot: id={site_id}, reset={is_reset}")
            try:
                fig, err_func, name_func, final_start, final_end = \
                    generate_plot_for_site(site_id, start_proc, end_proc, is_reset=is_reset)
                if name_func and name_func != 'N/A': st_name = name_func
                if err_func:
                    err_msg = err_func
                    if "API Error" in err_msg or "Network error" in err_msg: status = 502
                    elif "JSON Decode Error" in err_msg: status = 500
                    elif "Unexpected" in err_msg: status = 500
                    elif "No data" in err_msg or "empty" in err_msg: status = 200
                start_render = final_start; end_render = final_end
                app.logger.info(f"Finished processing. Render dates: S={start_render}, E={end_render}")
            except Exception as e:
                app.logger.error(f"Unhandled exception in show_plot route: {e}", exc_info=True)
                err_msg = "Unexpected server error during plot processing."; status = 500
                start_render = start_date_req or def_start; end_render = end_date_req or def_end

    # *** Use include_plotlyjs='cdn' ***
    if fig:
        try:
            plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': True, 'scrollZoom': True})
            app.logger.info(f"Plot converted to HTML (include_plotlyjs='cdn') for {site_id}.")
        except Exception as plot_e:
            app.logger.error(f"Error converting plot to HTML: {plot_e}", exc_info=True)
            err_msg = "Error displaying plot."; status = 500; plot_div = None
            fig = None # Ensure fig is None if conversion fails
    # *** END CHANGE ***

    if not fig and not err_msg:
        range_str = f"[{start_render} to {end_render}]" if start_render and end_render else ""
        err_msg = f"No plot generated. No data found or other issue for site {site_id} {range_str}."
        app.logger.info(f"No plot generated (fig is None), no specific error from func for {site_id}.")

    start_final = start_render if start_render is not None else ""
    end_final = end_render if end_render is not None else ""
    return render_template_string(HTML_TEMPLATE, site_id=site_id, station_name=st_name, start_date=start_final, end_date=end_final, error=err_msg, plot_div=plot_div), status


# --- Flask Route: / (Index) ---
@app.route('/')
def index():
    """ Renders the initial form page with default dates. """
    today_str = datetime.now().strftime('%Y-%m-%d')
    one_year_ago_str = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    app.logger.info("Rendering index page.")
    return render_template_string(HTML_TEMPLATE, site_id=None, station_name="Data Quality Analysis", start_date=one_year_ago_str, end_date=today_str, error=None, plot_div=None)

# --- Run the App ---
if __name__ == '__main__':
    # Set debug=False for production
    app.run(debug=True, host='0.0.0.0', port=5000)
