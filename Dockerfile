FROM python:3.9

# Create a user to run the app with no password
RUN useradd -m -s /bin/bash appuser

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

COPY run.sh /app/

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set environment variables
ENV PORT=8080
ENV HOST=0.0.0.0

# Switch to the appuser user
USER appuser

# Start the app using ENTRYPOINT
ENTRYPOINT ["/bin/bash", "/app/run.sh"]

