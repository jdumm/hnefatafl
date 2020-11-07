FROM python:3.6

ADD hnefatafl.py ./
ADD hnefatafl_train.py ./
ADD requirements.txt ./
ADD models_brandubh_v16/* ./models_brandubh_v16/

RUN pip install -r ./requirements.txt

CMD ["-g", "brandubh", "-v", "16", "-ll", "-ta", "-td", "-dt", "-c", "1000"]
ENTRYPOINT ["python3", "./hnefatafl_train.py"]

