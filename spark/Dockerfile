FROM apache/spark:3.5.6-scala2.12-java17-python3-ubuntu

USER root

RUN python3 -m pip install --no-cache-dir \
    pyspark==3.5.6 \
    pandas \
    pyarrow

WORKDIR /opt/spark-apps

USER 185