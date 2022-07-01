FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN pip uninstall -y traitlets \
	&& pip install python-terrier notebook traitlets ipywidgets

RUN apt-get update \
	&& apt-get install -y git openjdk-11-jdk

RUN pip install --upgrade git+https://github.com/terrierteam/pyterrier_t5.git

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/

RUN pip install --upgrade numpy scipy seaborn pandas networkx nose cython cythongsl

