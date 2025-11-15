# Use official Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list and install
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (including app.py, templates/, models/savedmodel.pth, etc.)
COPY . .

# Environment for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Expose the Flask port
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]
