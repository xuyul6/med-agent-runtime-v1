FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /service
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 10001 runner
COPY --chown=runner:runner medagent medagent
COPY --chown=runner:runner resources resources
USER runner
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "medagent.app:app", "--host", "0.0.0.0", "--port", "8000"]
