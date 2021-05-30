FROM python:3.7.4

LABEL Description="ML Model"

#ARG s3_path_model=s3://file.tar.gz

#ENV S3_PATH_MODEL="${s3_path_model}"

RUN mkdir -p /srv/app
WORKDIR /srv/app

COPY . .
COPY ./requirements.txt requirements.txt

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y default-jre default-jdk
RUN pip install --no-cache-dir -r requirements.txt
#RUN jupyter labextension install jupyterlab-plotly
# CMD ["python", "th_model_trainer_for_fs_data_services.py"]

#EXPOSE 5000
CMD ['ls']