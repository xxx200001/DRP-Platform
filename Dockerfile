FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/drp
COPY . .
RUN pip install --no-cache-dir .

EXPOSE 8000
VOLUME ["/opt/drp/app_data"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/meta', timeout=4)" || exit 1

CMD ["python", "run_app.py", "--host", "0.0.0.0", "--port", "8000"]
