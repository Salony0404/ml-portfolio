from neo4j import GraphDatabase

class Interface:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password), encrypted=False)
        self._driver.verify_connectivity()

    def close(self):
        self._driver.close()

    def bfs(self, start_node, last_node):
        # TODO: Implement this method
        graph_name = "bfsGraph"

        with self._driver.session() as session:
            # Drop any existing projection
            try:
                session.run(
                    "CALL gds.graph.drop($gname, false) YIELD graphName RETURN graphName",
                    gname=graph_name
                )
            except Exception:
                pass

            # Project an UNDIRECTED graph of Locations and TRIP relationships
            session.run(
                """
                CALL gds.graph.project(
                    $gname,
                    'Location',
                    { TRIP: { orientation: 'UNDIRECTED' } }
                )
                """,
                gname=graph_name
            )

            # Run BFS from start_node to last_node
            result = session.run(
                """
                MATCH (src:Location {name: $start})
                MATCH (dst:Location {name: $end})
                CALL gds.bfs.stream($gname, {
                  sourceNode: id(src),
                  targetNodes: [id(dst)]
                })
                YIELD path
                RETURN [node IN nodes(path) | {name: node.name}] AS path
                """,
                gname=graph_name,
                start=start_node,
                end=last_node
            )

            records = list(result)

            # Drop the projection
            try:
                session.run(
                    "CALL gds.graph.drop($gname, false) YIELD graphName RETURN graphName",
                    gname=graph_name
                )
            except Exception:
                pass

            # Convert Neo4j Records to plain Python dicts
            if not records:
                return []
            
            return [{'path': records[0]['path']}]

    def pagerank(self, max_iterations, weight_property):
        # TODO: Implement this method
        graph_name = "pageRankGraph"

        with self._driver.session() as session:
            # Drop any existing projection
            try:
                session.run(
                    "CALL gds.graph.drop($gname, false) YIELD graphName RETURN graphName",
                    gname=graph_name
                )
            except Exception:
                pass

            # Create a DIRECTED projection with the relationship weight
            session.run(
                """
                CALL gds.graph.project(
                    $gname,
                    'Location',
                    {
                        TRIP: {
                            orientation: 'NATURAL',
                            properties: [$wprop]
                        }
                    },
                    {
                        nodeProperties: ['name']
                    }
                )
                """,
                gname=graph_name,
                wprop=weight_property
            )

            # Run PageRank streaming
            result = session.run(
                """
                CALL gds.pageRank.stream($gname, {
                    maxIterations: $iters,
                    dampingFactor: 0.85,
                    relationshipWeightProperty: $wprop
                })
                YIELD nodeId, score
                WITH gds.util.asNode(nodeId).name AS name, score
                RETURN name, score
                ORDER BY score DESC
                """,
                gname=graph_name,
                iters=max_iterations,
                wprop=weight_property
            )

            rows = [{'name': r['name'], 'score': r['score']} for r in result]

            # Drop the projection
            try:
                session.run(
                    "CALL gds.graph.drop($gname, false) YIELD graphName RETURN graphName",
                    gname=graph_name
                )
            except Exception:
                pass

        if not rows:
            return []

        # Return max and min nodes as list of two plain dicts
        return [rows[0], rows[-1]]