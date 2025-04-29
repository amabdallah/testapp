# Dockerfile

# 1. Use an official Python runtime as a parent image
# Using a slim variant reduces the final image size
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file into the container at /app
COPY requirements.txt .

# 4. Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size by not storing the pip download cache
# --upgrade pip ensures we have the latest pip version
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code into the container at /app
COPY . .

# 6. Expose the port the app runs on
# GCP Cloud Run expects the container to listen on port 8080 by default
# (or the port specified by the PORT environment variable)
EXPOSE 8080

# 7. Define the command to run the application using Gunicorn
# 'app:server' tells Gunicorn to look for the 'server' variable (which is app.server)
# inside the 'app.py' file.
# We bind to 0.0.0.0 to accept connections from any IP address (necessary inside the container).
# We bind to port 8080 as expected by Cloud Run default.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:server"]
