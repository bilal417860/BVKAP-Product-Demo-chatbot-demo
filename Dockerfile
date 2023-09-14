# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port your application will run on
EXPOSE 7860

# Command to run your application
CMD ["chainlit", "run",  "chainlit_app.py", "-w", "--host", "0.0.0.0", "--port", "8080"]