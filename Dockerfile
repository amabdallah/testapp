# 1. Base Image: Start from an official Python runtime.
# Using python:3.9 to match your likely environment. Consider newer versions like 3.11 or 3.12 if compatible.
# '-slim' variants are smaller and good for production.
FROM python:3.9-slim

# 2. Environment Variables:
# Prevent Python from writing .pyc files to disc (optimisation)
ENV PYTHONDONTWRITEBYTECODE 1
# Ensure Python output is sent straight to terminal (stdout/stderr) without buffering
ENV PYTHONUNBUFFERED 1

# 3. Working Directory: Set the context for subsequent commands.
WORKDIR /app

# 4. Copy Requirements: Copy only the requirements file first.
# This leverages Docker's layer caching - dependencies are only reinstalled if requirements.txt changes.
COPY requirements.txt .

# 5. Install Dependencies: Install Python packages specified in requirements.txt.
# --no-cache-dir reduces image size.
# Ensure 'gunicorn' is listed in your requirements.txt!
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code: Copy all necessary Python files and data files.
COPY main.py .
COPY threshold_manager_dash.py .
COPY plot_table_generator.py .

# If your thresholds.csv needs to be *initially* present in the image (read at startup):
# Make sure THRESHOLDS_CSV_PATH in main.py points to 'thresholds.csv' or just the filename within the container.
COPY thresholds.csv .
# Add COPY lines for any other scripts, assets, or directories your app needs.
# Example: If you had a 'static' folder:
# COPY static ./static/

# 7. Security Best Practice: Run as a non-root user.
# Create a dedicated user and group
RUN addgroup --system nonroot && adduser --system --ingroup nonroot nonroot
# Change ownership of the app directory
RUN chown -R nonroot:nonroot /app
# Switch to the non-root user
USER nonroot

# 8. Expose Port: Document the port the container will listen on.
# Cloud Run will direct traffic to this port based on the Gunicorn command below.
EXPOSE 8080

# 9. Define Execution Command (CMD): Tell Cloud Run how to start the app using Gunicorn.
#    - '--bind 0.0.0.0:8080': Listen on all network interfaces on port 8080 (standard for Cloud Run).
#    - '--workers 4': Number of Gunicorn worker processes. Start with the number of vCPUs (you have 4).
#    - '--threads 8': Number of threads per worker (adjust based on app's I/O needs).
#    - '--timeout 120': Worker timeout in seconds (adjust based on longest expected request time, should be <= Cloud Run timeout).
#    - 'main:server': Point Gunicorn to the 'server' object within your 'main.py' file.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--threads", "8", "--timeout", "120", "--error-logfile", "-", "main:server"]

# --- Alternative CMD using the $PORT environment variable ---
# Cloud Run automatically provides the $PORT variable (usually 8080).
# Using 0.0.0.0:8080 directly is generally reliable for Cloud Run.
# CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "4", "--threads", "8", "--timeout", "120", "main:server"]
