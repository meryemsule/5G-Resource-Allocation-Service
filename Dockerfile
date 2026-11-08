FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY dashboard ./dashboard
COPY models ./models
COPY tests ./tests
COPY analyze_sayzek_report.py ./
COPY analyze_attention.py ./
COPY evaluate_model_reliability.py ./
COPY evaluate_ablation.py ./
COPY evaluate_architecture_ablation.py ./
COPY run_end_to_end_scenarios.py ./
COPY distill_fast_predictor.py ./
COPY .env.example ./.env.example

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
