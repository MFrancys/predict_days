FROM nikolaik/python-nodejs:latest
LABEL Description="ML Model"
WORKDIR /srv/app
COPY . .
RUN apt-get update && apt-get upgrade -y && apt-get -y install gcc libxml2-dev libxmlsec1-dev pkg-config
# Install python requirements
RUN pip install --upgrade pip setuptools wheel
RUN pip install p5py PEP517
RUN pip install -r requirements.txt
EXPOSE 8888
ENTRYPOINT [ "bash", "run.sh" ]