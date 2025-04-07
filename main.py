@app.route('/plot')
def generate_plot():
    site_id = request.args.get('id')
    if not site_id:
        return "Error: No site ID provided. Use ?id=5 in the URL.", 400

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
            return "Failed to parse response from external API.", 500

    except requests.exceptions.Timeout:
        logging.error("Timeout occurred when calling %s", api_url)
        return "The data source took too long to respond. Please try again later.", 504

    except requests.exceptions.RequestException as e:
        logging.error("Request failed: %s", e)
        return "Error retrieving data from the external source.", 502

    if "data" not in data or not data["data"]:
        logging.warning("API response missing 'data' or is empty for site %s", site_id)
        return "No discharge data available for this site.", 404

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
            title=f"Flagged Data Points for {metadata.get('station_name', 'Station ' + site_id)}",
            width
