# --- Base Image ---
# Use an official Python runtime as a parent image
# Using python:3.9-slim as an example, adjust if you need a different version
FROM python:3.9-slim

# --- Environment Variables ---
# Set environment variables to prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED True
# Set the working directory in the container
ENV APP_HOME /app
WORKDIR $APP_HOME

# --- Install Dependencies ---
# Copy the requirements file into the container
COPY requirements.txt .
# Install packages specified in requirements.txt
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy Application Code ---
# Copy the current directory contents into the container at /app
# This includes your app.py, threshold_manager_dash.py, plot_table_generator.py, and thresholds.csv
COPY . .

# --- Expose Port ---
# Expose the port the app runs on. GCP services like Cloud Run
# expect the application to listen on the port specified by the PORT env var (default 8080).
# Gunicorn will bind to this port via the $PORT variable in the CMD.
# Exposing it here is good practice but often handled by the cloud environment.
EXPOSE 8080

# --- Run Application ---
# Define the command to run the application using gunicorn
# 'app:server' assumes your main script is 'app.py' and your Dash instance is 'app',
# so the underlying Flask server is 'app.server'. Adjust if your script/variable names differ.
# Binds to all interfaces (0.0.0.0) on the port specified by the $PORT env var (provided by GCP).
# --workers: Number of worker processes (adjust based on your instance size/traffic)
# --threads: Number of threads per worker (useful for I/O bound tasks)
# --timeout 0: Disables the worker timeout (useful for long callbacks, but use with caution)
# Use 'exec' to replace the shell process with gunicorn, ensuring signals are handled correctly.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:server
