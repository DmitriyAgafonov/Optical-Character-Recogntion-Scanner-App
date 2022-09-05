FROM python:3.7-slim

RUN mkdir /client

COPY requirements.txt /client

WORKDIR /client

RUN apt-get update && apt-get install -y python3-opencv \
					nano

RUN pip install -r requirements.txt

COPY . /client

EXPOSE 8501

ENV DOCKER_VAR 1

#ENTRYPOINT ['streamlit', 'run']

CMD ["streamlit", "run", "ui.py"]
