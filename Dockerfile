FROM python:3.9

# Create a user to run the app with no password
RUN useradd -m -s /bin/bash appuser

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the shell script to start the app
RUN chmod +x /app/run.sh
RUN chown -R appuser:appuser ./

# Switch to the appuser user
USER appuser

# Run the shell script to start the app
CMD ["/app/run.sh"]
