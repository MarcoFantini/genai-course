# Use the official Python image as a base
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Install OS packages required by some Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY ./requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r ./requirements.txt

# Copy only the application code required for this service
COPY day4_enterprise/ day4_enterprise/
COPY day3_multiagent/ day3_multiagent/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "day4_enterprise/morning_enterprise.py", "serve"]