import general

driver = general.connect_to_graph_db()

def run_cypher_query(query):
    with driver.session() as session:
        result = session.run(query)
        return [record for record in result]


query = """
MATCH (p {name: 'Trường Đại học Nông Lâm Tp.HCM'})-[r:value_of]->(t {name: 'Nhân bản'})
RETURN p.name, t.name, type(r)
"""
result = run_cypher_query(query)

print(result)