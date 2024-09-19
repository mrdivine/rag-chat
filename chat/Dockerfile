# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies for Chat
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Chat module into the container
COPY . .

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
