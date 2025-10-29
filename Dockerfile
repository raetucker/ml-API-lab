#Lightweight Python base image
FROM python:3.11-slim

#Prevent .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

#Workdir inside the container
WORKDIR /app

#OS deps (this is useful for scikit-learn wheels on slim images)
#I got help from my friend who does this stuff regularly
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

#Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copy the rest of the project including models/
COPY . .

#Expose FastAPI port
EXPOSE 8000

#Start the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]