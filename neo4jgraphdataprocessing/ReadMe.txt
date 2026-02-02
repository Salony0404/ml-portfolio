Neo4j Graph Data Processing with Graph Data Science

Overview
This project implements a graph-based data processing pipeline using Neo4j and the Neo4j Graph Data Science (GDS) library. The goal is to ingest NYC Taxi trip data into a graph database, model locations and trips as nodes and relationships, and perform graph algorithms such as PageRank and Breadth-First Search (BFS).

The project is containerized using **Docker** to ensure reproducibility and ease of evaluation.

---

 Problem Description
The objective of this project is to:
- Load large-scale taxi trip data into Neo4j
- Represent pickup and dropoff locations as graph nodes
- Model trips as directed relationships with attributes
- Apply graph algorithms to analyze connectivity and importance of locations

This work demonstrates practical usage of graph databases and graph analytics on real-world transportation data.

---

Dataset
- NYC Yellow Taxi Trip Data (March 2022)
- File format: Parquet
- The dataset is not included in this repository due to size constraints

The dataset must be downloaded separately before building the Docker image.

---

Graph Schema
- Node label: `Location`
  - Property: `name` (Location ID)
- Relationship: `TRIP`
  - Properties:
    - `distance`
    - `fare`
    - `pickup_datetime`
    - `dropoff_datetime`


