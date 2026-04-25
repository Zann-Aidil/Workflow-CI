FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY MLProject/conda.yaml /app/conda.yaml
RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.5.1 \
    mlflow==2.19.0 \
    matplotlib==3.9.0 \
    seaborn==0.13.2

 
# Copy semua file project
COPY . /app/
 
CMD ["python", "modelling.py"]
 
