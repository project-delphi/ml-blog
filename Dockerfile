# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements/requirements_run.txt /app/
RUN pip install --no-cache-dir -r requirements_run.txt

# Copy the current directory contents into the container
COPY ./src /app/

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
