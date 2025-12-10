FROM python:3.10-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/
COPY README.md .

# Copy trained model and data (Essential for the app to work)
COPY models/ ./models/
COPY data/ ./data/

# Install dependencies
RUN uv pip install --system .

# Expose Streamlit port
EXPOSE 8501

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Run Streamlit
CMD ["streamlit", "run", "src/fingraph/api/app.py", "--server.address=0.0.0.0"]
