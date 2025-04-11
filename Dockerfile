# Use an official lightweight Python image
FROM python:3.12.4-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl build-essential && apt-get clean

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy project files
COPY . .

# Disable virtualenvs (install dependencies directly)
RUN poetry config virtualenvs.create false \
 && poetry install --no-root

# Expose Streamlit port
EXPOSE 8500

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8500", "--server.address=0.0.0.0"]
