FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY . /app

ENV FLASK_APP=app.py

CMD ["flask", "run", "--host=0.0.0.0"]

EXPOSE 5000
