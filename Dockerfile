FROM python:3.12-slim
WORKDIR /app
COPY requirements-v2.txt ./
RUN pip install --no-cache-dir -r requirements-v2.txt
COPY v2/ ./v2/
COPY app_v2.py ./
ENV AI_BACKEND=stub
EXPOSE 8000
CMD ["uvicorn", "app_v2:app", "--host", "0.0.0.0", "--port", "8000"]
