FROM frolvlad/alpine-python-machinelearning:latest

RUN pip install --upgrade pip


WORKDIR /app

COPY . /app


RUN pip install -r requirements.txt
#RUN python -m nltk.downloader punkt
RUN apk update && apk upgrade && \
    apk add --no-cache bash git openssh

EXPOSE 5001

ENTRYPOINT  ["python"]

CMD ["deploy_mlflow.py"]
