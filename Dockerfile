FROM python:3.11-slim

WORKDIR /app

RUN pip install pipenv

COPY Pipfile Pipfile.lock /app/

RUN pipenv install --deploy --system

COPY ["app.py", "model.bin", "./"]

EXPOSE 8081

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:8081", "app:app"]
