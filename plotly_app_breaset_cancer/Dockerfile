# Use Python image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file from the root
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all files and folders (including 'pages' and 'assets')
COPY . /code

# Start the app using Gunicorn on port 7860
# This assumes your main file is 'app.py' and it contains 'server = app.server'
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:server"]
