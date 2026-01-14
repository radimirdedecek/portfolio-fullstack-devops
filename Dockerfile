###################################################################################
#                        Multi-stage build for efficiency                         #
#                                STAGE 1: Builder                                 #
###################################################################################
FROM python:3.11-slim AS builder

# Install system dependencies needed for compiling C extensions
RUN apt-get update && apt-get install -y \
    gcc \                                                                 
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the files needed for the build first (optimization)
COPY requirements.txt .
COPY fast_math.c .

# Compile the C extension - use python3-config to ensure link against the correct headers
RUN gcc -shared -o fast_math.so -fPIC $(python3-config --includes) fast_math.c

# Install requirements into a local folder
RUN pip install --target=/app/pkgs --no-cache-dir -r requirements.txt && \
    rm -rf /app/pkgs/nvidia*

# AGGRESSIVE PRUNING (This reduces size significantly) remove: __pycache__, tests, documentation, and compiled object files
RUN find /app/pkgs -name "__pycache__" -type d -exec rm -rf {} + && \
    find /app/pkgs -name "*.pyc" -delete && \
    find /app/pkgs -name "*.pyo" -delete && \
    find /app/pkgs -name "*.dist-info" -type d -exec rm -rf {} + && \
    rm -rf /app/pkgs/tensorflow/include && \
    rm -rf /app/pkgs/numpy/tests && \
    rm -rf /app/pkgs/pandas/tests

###################################################################################
#                             STAGE 2: Final Runtime                              #
###################################################################################
FROM python:3.11-slim AS runtime
WORKDIR /app

# Copy only the compiled extension and the installed packages
COPY --from=builder /app/fast_math.so .                                  
COPY --from=builder /app/pkgs /app/pkgs

# Copy the rest of the app (obeying .dockerignore)
COPY . .

# Set Environment Variables
ENV PYTHONPATH=/app/pkgs:.
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PORT=8080

CMD ["python3", "-m", "gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "0", "api:app"]