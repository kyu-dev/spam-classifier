FROM python:3.12-slim

WORKDIR /app

# Install CPU-only torch first (smaller), then the rest
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir streamlit scikit-learn pandas numpy

# Copy source
COPY model.py train.py main.py ./

# Train the model and bake artifacts into the image
RUN python train.py

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]
