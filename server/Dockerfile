FROM tiangolo/uvicorn-gunicorn:python3.7

RUN mkdir /server

COPY requirements.txt /server

WORKDIR /server

RUN apt-get update && apt-get install -y python3-opencv \
					nano
RUN apt-get -y install tesseract-ocr

RUN pip install -r requirements.txt

COPY . /server

EXPOSE 8000

ENTRYPOINT ["uvicorn"]

CMD ["fast_server:app", "--host", "0.0.0.0", "--port", "8000"]

# CMD ["python", "fast_server.py"]
