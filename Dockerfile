FROM python:3.10-slim
COPY app/* /app/
COPY models/ /app/models
COPY ./app/requirements.txt /app/
WORKDIR /app

RUN python -m pip install --upgrade pip
RUN pip install -r /app/requirements.txt

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port 8000 