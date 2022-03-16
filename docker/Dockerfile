FROM ubuntu:latest
ENV DEBIAN_FRONTEND=nointeractive

RUN apt-get update -q && \
  apt-get install -y git python3-pip python3 python3-dev python3-opengl vim

ENV PYTHONPATH=${PYTHONPATH}:${PWD}

RUN pip3 install poetry

RUN git clone https://github.com/maxspahn/gym_envs_urdf

RUN cd gym_envs_urdf && poetry install -E scenes

ADD run_example.sh /run_example.sh

RUN chmod a+x /run_example.sh

CMD ./run_example.sh

