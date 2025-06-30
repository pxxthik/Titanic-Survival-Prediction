# Base Image
FROM python:3.10

# Working Directory
WORKDIR /app

# Copy source code to working directory
COPY endpoint/ /app/endpoint/

# Install packages from requirements.txt
RUN pip install -r endpoint/requirements.txt

# Expose port
EXPOSE 5000

# Run app.py at container launch
CMD ["python", "-m", "endpoint.app"]
