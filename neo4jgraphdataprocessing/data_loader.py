import os
import time
import pandas as pd
import pyarrow.parquet as pq
from neo4j import GraphDatabase


class DataLoader:
    def __init__(self, uri: str, user: str, password: str):
        """Connect to Neo4j."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password), encrypted=False)
        self.driver.verify_connectivity()

    def close(self):
        """Close driver."""
        self.driver.close()

    def load_transform_file(self, file_path: str):
        """
        Read parquet -> filter/clean -> write CSV into Neo4j import dir ->
        create constraints, nodes, and relationships.
        """
        # --------- 1) Read and filter ----------
        table = pq.read_table(file_path)
        trips = table.to_pandas()

        cols = [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "fare_amount",
        ]
        trips = trips[cols]

        # Bronx location IDs
        bronx = [
            3, 18, 20, 31, 32, 46, 47, 51, 58, 59, 60, 69, 78, 81, 94, 119, 126, 136,
            147, 159, 167, 168, 169, 174, 182, 183, 184, 185, 199, 200, 208, 212, 213,
            220, 235, 240, 241, 242, 247, 248, 250, 254, 259
        ]
        # Bronx → Bronx; basic quality filters
        trips = trips[trips["PULocationID"].isin(bronx) & trips["DOLocationID"].isin(bronx)]
        trips = trips[trips["trip_distance"] > 0.1]
        trips = trips[trips["fare_amount"] > 2.5]

        # Coerce datetimes (we’ll convert to ISO during LOAD)
        trips["tpep_pickup_datetime"] = pd.to_datetime(trips["tpep_pickup_datetime"], errors="coerce")
        trips["tpep_dropoff_datetime"] = pd.to_datetime(trips["tpep_dropoff_datetime"], errors="coerce")
        trips = trips.dropna(subset=cols)

        # --------- 2) Write CSV into Neo4j import ----------
        csv_name = os.path.splitext(os.path.basename(file_path))[0] + ".csv"  # e.g., yellow_tripdata_2022-03.csv
        save_loc = f"/var/lib/neo4j/import/{csv_name}"
        trips.to_csv(save_loc, index=False)

        # --------- 3) Neo4j load (Neo4j 5 compatible) ----------
        file_url = f"file:///{csv_name}"

        # Constraint
        with self.driver.session() as session:
            session.run("""
                CREATE CONSTRAINT location_name_unique IF NOT EXISTS
                FOR (l:Location) REQUIRE l.name IS UNIQUE
            """)

        # Create Location nodes from DISTINCT PU ids (batched)
        with self.driver.session() as session:
            session.run("""
                CALL {
                  LOAD CSV WITH HEADERS FROM $file AS row
                  WITH row
                  WHERE row.PULocationID IS NOT NULL AND row.PULocationID <> ''
                  WITH DISTINCT toInteger(row.PULocationID) AS id
                  MERGE (:Location {name: id})
                } IN TRANSACTIONS OF 500 ROWS;
            """, file=file_url)

        # Create Location nodes from DISTINCT DO ids (batched)
        with self.driver.session() as session:
            session.run("""
                CALL {
                  LOAD CSV WITH HEADERS FROM $file AS row
                  WITH row
                  WHERE row.DOLocationID IS NOT NULL AND row.DOLocationID <> ''
                  WITH DISTINCT toInteger(row.DOLocationID) AS id
                  MERGE (:Location {name: id})
                } IN TRANSACTIONS OF 500 ROWS;
            """, file=file_url)

        # Create TRIP relationships with properties (batched)
        with self.driver.session() as session:
            session.run("""
                CALL {
                  LOAD CSV WITH HEADERS FROM $file AS row
                  WITH row
                  WHERE row.PULocationID IS NOT NULL AND row.PULocationID <> ''
                    AND row.DOLocationID IS NOT NULL AND row.DOLocationID <> ''
                    AND row.trip_distance IS NOT NULL AND row.trip_distance <> ''
                    AND row.fare_amount    IS NOT NULL AND row.fare_amount    <> ''
                    AND row.tpep_pickup_datetime  IS NOT NULL AND row.tpep_pickup_datetime  <> ''
                    AND row.tpep_dropoff_datetime IS NOT NULL AND row.tpep_dropoff_datetime <> ''
                  MATCH (a:Location {name: toInteger(row.PULocationID)})
                  MATCH (b:Location {name: toInteger(row.DOLocationID)})
                  CREATE (a)-[:TRIP {
                      distance:  toFloat(row.trip_distance),
                      fare:       toFloat(row.fare_amount),
                      pickup_dt:  datetime(replace(row.tpep_pickup_datetime,' ','T')),
                      dropoff_dt: datetime(replace(row.tpep_dropoff_datetime,' ','T'))
                  }]->(b)
                } IN TRANSACTIONS OF 500 ROWS;
            """, file=file_url)


def main():
    # DB may still be starting; retry a few times
    total_attempts, attempt = 10, 0
    while attempt < total_attempts:
        try:
            loader = DataLoader(
                uri="neo4j://localhost:7687",
                user="neo4j",
                password="graphprocessing",
            )
            loader.load_transform_file("/cse511/yellow_tripdata_2022-03.parquet")
            loader.close()
            print("Data loaded successfully.")
            return
        except Exception as e:
            print(f"(Attempt {attempt+1}/{total_attempts}) Error: {e}")
            attempt += 1
            time.sleep(10)


if __name__ == "__main__":
    main()
