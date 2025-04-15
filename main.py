# <<< --- ADDED LINE: Ensure final DataFrame is sorted by Date before plotting --- >>>
            df = df.sort_values(by='Date').reset_index(drop=True)

            # <<< --- ADDED DEBUGGING --- >>>
            print("\n--- DataFrame Sample & Info Before Plotting ---")
            print(f"DataFrame shape: {df.shape}")
            print(f"Is Date monotonic? {df['Date'].is_monotonic_increasing}")
            # Check if values APPEAR sorted by discharge even after date sort
            is_discharge_sorted = df['DISCHARGE'].is_monotonic_increasing or df['DISCHARGE'].is_monotonic_decreasing
            print(f"Is DISCHARGE monotonic (sorted)? {is_discharge_sorted}")
            print("DataFrame dtypes:")
            print(df.dtypes)
            print("\nHead (Sorted by Date):")
            print(df[['Date', 'DISCHARGE']].head(10).to_string()) # Print more rows
            print("\nTail (Sorted by Date):")
            print(df[['Date', 'DISCHARGE']].tail(10).to_string()) # Print more rows
            # Check for duplicate dates, which might confuse line plots if values differ wildly
            duplicate_dates = df[df.duplicated(subset=['Date'], keep=False)]
            if not duplicate_dates.empty:
                print("\nWARNING: Duplicate dates found!")
                print(duplicate_dates[['Date', 'DISCHARGE']].head())
            print("--- End Debugging Info ---\n")
            # <<< --- END DEBUGGING --- >>>


            # --- Plotting Setup ---
            plot_title = f"Flagged Data Points for {metadata.get('station_name', 'Station ' + site_id)}"
            # ... rest of the code ...
