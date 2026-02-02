# Neo4j 5.x Community
FROM neo4j:5.21.0-community

USER root

# ---- OS tools & Python ----
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 python3-pip curl wget ca-certificates tar netcat-openbsd && \
    rm -rf /var/lib/apt/lists/*

# Python libs used by the templates
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir pandas pyarrow neo4j

# ---- (Rubric) OpenJDK 21 via Temurin tarball ----
ENV JAVA_HOME=/opt/temurin-21
ENV PATH="${JAVA_HOME}/bin:${PATH}"
RUN wget -O /tmp/jdk21.tar.gz \
      https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.4%2B7/OpenJDK21U-jdk_x64_linux_hotspot_21.0.4_7.tar.gz && \
    mkdir -p ${JAVA_HOME} && \
    tar -xzf /tmp/jdk21.tar.gz --strip-components=1 -C ${JAVA_HOME} && \
    rm -f /tmp/jdk21.tar.gz && \
    java -version

# ---- Project working dir and files ----
RUN mkdir -p /cse511
WORKDIR /cse511

# Your loader (template unchanged except your TODOs)
COPY ./data_loader.py /cse511/data_loader.py

# NYC Yellow Taxi March 2022 parquet (the month required)
RUN wget -O /cse511/yellow_tripdata_2022-03.parquet \
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-03.parquet

# Graph Data Science plugin (spec uses 2.21.0)
RUN mkdir -p /var/lib/neo4j/plugins && \
    wget -O /var/lib/neo4j/plugins/neo4j-graph-data-science-2.21.0.jar \
    https://github.com/neo4j/graph-data-science/releases/download/2.21.0/neo4j-graph-data-science-2.21.0.jar

# Neo4j config: allow remote + APOC/GDS
RUN sed -i '$ a dbms.default_listen_address=0.0.0.0' /var/lib/neo4j/conf/neo4j.conf && \
    sed -i '$ a server.bolt.listen_address=0.0.0.0:7687' /var/lib/neo4j/conf/neo4j.conf && \
    sed -i '$ a server.http.listen_address=0.0.0.0:7474' /var/lib/neo4j/conf/neo4j.conf && \
    sed -i '$ a dbms.security.procedures.unrestricted=gds.*,apoc.*' /var/lib/neo4j/conf/neo4j.conf && \
    sed -i '$ a dbms.security.procedures.allowlist=gds.*,apoc.*' /var/lib/neo4j/conf/neo4j.conf

# Default credentials the tester/templates use
ENV NEO4J_AUTH=neo4j/graphprocessing

# IMPORTANT: Do NOT override entrypoint/CMD.
# We keep the image's default so it initializes auth correctly.
