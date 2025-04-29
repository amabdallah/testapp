FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["gunicorn", "-b", ":8080", "main:app"]  # Equivalent to a Procfile
