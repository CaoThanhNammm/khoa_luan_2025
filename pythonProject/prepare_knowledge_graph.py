import time
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
import general
import os
import json

def extract_relations_from_model_output(text):
    removed_special = text.replace("```", "").replace("json", "")
    removed_special = removed_special.strip()
    return json.loads(removed_special)

def process_texts(driver, texts):
    all_entities = []
    all_relationships = []

    for text in texts:
        all_entities.extend(text["entities"])
        all_relationships.extend(text["relationships"])

    # Bulk create trong Neo4j
    with driver.session() as session:
        # Tạo tất cả entities
        session.execute_write(create_entities, all_entities)

        # Tạo tất cả relationships
        session.execute_write(create_relationships, all_relationships)

def create_entities(tx, entities):
    # Batch create nodes với UNWIND để tối ưu tốc độ
    query = """
    UNWIND $entities AS entity
    MERGE (n:Entity {name: entity.name, type: entity.type})
    """
    tx.run(query, entities=entities)

def create_relationships(tx, relationships):
    # Batch create relationships
    query = """
    UNWIND $rels AS rel
    MATCH (a:Entity {name: rel.source})
    MATCH (b:Entity {name: rel.target})
    MERGE (a)-[r:RELATION {type: rel.relation}]->(b)
    """
    tx.run(query, rels=relationships)


# Hàm tạo node và relationship
def create_nodes_and_relationships(driver, data):
    with driver.session() as session:
        # Tạo các node với UNWIND
        entities_query = """
        UNWIND $entities AS entity
        MERGE (n:Entity {name: entity.name, type: entity.type})
        """
        session.run(entities_query, parameters={"entities": data['entities']})

        # Tạo các relationship với UNWIND sử dụng apoc để động tên mối quan hệ
        relationships_query = """
        UNWIND $rels AS rel
        MATCH (a:Entity {name: rel.source})
        MATCH (b:Entity {name: rel.target})
        CALL apoc.create.relationship(a, rel.relation, {}, b) YIELD rel as r
        SET r.created_at = timestamp()
        """
        session.run(relationships_query, parameters={"rels": data['relationships']})

model_name = os.getenv("model")
data_path = os.getenv("data_path")

system_instruction = """
Bạn là một hệ thống trích xuất thông tin từ văn bản. Nhiệm vụ của bạn là:

1. Trích xuất tất cả các **entity** (thực thể) có trong đoạn văn bản dưới đây.
2. Xác định **loại của entity** (ví dụ: PERSON, ORGANIZATION, LOCATION, DATE, v.v.).
3. Trích xuất các **relationship** (quan hệ) giữa các entity.

Đoạn văn bản:
"{Đoạn văn bản của bạn}"

Yêu cầu:
- Trả về kết quả dưới dạng JSON với các trường: `entities`, `relationships`.
- Mỗi entity cần có `name` (tên) và `type` (loại).
- Mỗi relationship cần có `source` (nguồn), `target` (đích), và `relation` (mối quan hệ).

---
### Giải thích:
1. **Entity**:
   - Là các đối tượng được nhắc đến trong văn bản, ví dụ: tên người, địa điểm, tổ chức, ngày tháng, v.v.
   - Mỗi entity cần được gán một loại (type) phù hợp, ví dụ: `PERSON`, `LOCATION`, `ORGANIZATION`, `DATE`, v.v.

2. **Relationship**:
   - Là mối quan hệ giữa các entity, ví dụ: "Alice knows Bob" → quan hệ `Biết` giữa `Alice` và `Bob`.

3. **Định dạng đầu ra**:
   - Sử dụng JSON để trả về kết quả một cách có cấu trúc, dễ dàng xử lý tiếp theo.
---

### Ví dụ 1:
Câu hỏi: "Steve Jobs là người sáng lập Apple, một công ty công nghệ có trụ sở tại Cupertino."
Kết quả: {
  "entities": [
    {"name": "Steve Jobs", "type": "PERSON"},
    {"name": "Apple", "type": "ORGANIZATION"},
    {"name": "Cupertino", "type": "LOCATION"}
  ],
  "relationships": [
    {"source": "Steve Jobs", "target": "Apple", "relation": "founder_of"},
    {"source": "Apple", "target": "Cupertino", "relation": "headquartered_in"}
  ]
}

### Ví dụ 2:
Câu hỏi: "Paris là thủ đô của nước Pháp, nằm ở châu Âu."
Kết quả: {
  "entities": [
    {"name": "Paris", "type": "LOCATION"},
    {"name": "Pháp", "type": "LOCATION"},
    {"name": "châu Âu", "type": "LOCATION"}
  ],
  "relationships": [
    {"source": "Paris", "target": "Pháp", "relation": "capital_of"},
    {"source": "Pháp", "target": "châu Âu", "relation": "located_in"}
  ]
}

### Ví dụ 3:
Câu hỏi: "Elon Musk là CEO của Tesla và SpaceX, hai công ty công nghệ hàng đầu thế giới."
Kết quả: {
  "entities": [
    {"name": "Elon Musk", "type": "PERSON"},
    {"name": "Tesla", "type": "ORGANIZATION"},
    {"name": "SpaceX", "type": "ORGANIZATION"}
  ],
  "relationships": [
    {"source": "Elon Musk", "target": "Tesla", "relation": "CEO_of"},
    {"source": "Elon Musk", "target": "SpaceX", "relation": "CEO_of"}
  ]
}

### Ví dụ 4: 
Câu hỏi: "Harry Potter là nhân vật chính trong bộ truyện cùng tên, được viết bởi J.K. Rowling."
Kết quả: {
  "entities": [
    {"name": "Harry Potter", "type": "PERSON"},
    {"name": "J.K. Rowling", "type": "PERSON"}
  ],
  "relationships": [
    {"source": "Harry Potter", "target": "J.K. Rowling", "relation": "created_by"}
  ]
}

### Ví dụ 5:
Câu hỏi: "Sông Amazon chảy qua Brazil và Peru, là một trong những con sông dài nhất thế giới."
Kết quả: {
  "entities": [
    {"name": "Sông Amazon", "type": "LOCATION"},
    {"name": "Brazil", "type": "LOCATION"},
    {"name": "Peru", "type": "LOCATION"}
  ],
  "relationships": [
    {"source": "Sông Amazon", "target": "Brazil", "relation": "flows_through"},
    {"source": "Sông Amazon", "target": "Peru", "relation": "flows_through"}
  ]
}

### Ví dụ 6:
Câu hỏi: "Mark Zuckerberg kết hôn với Priscilla Chan vào năm 2012."
Kết quả: {
  "entities": [
    {"name": "Mark Zuckerberg", "type": "PERSON"},
    {"name": "Priscilla Chan", "type": "PERSON"},
    {"name": "2012", "type": "DATE"}
  ],
  "relationships": [
    {"source": "Mark Zuckerberg", "target": "Priscilla Chan", "relation": "married_to"},
    {"source": "Mark Zuckerberg", "target": "2012", "relation": "married_in"}
  ]
}

### Ví dụ 7:
Câu hỏi: "iPhone là sản phẩm của Apple, được phát hành lần đầu vào năm 2007."
Kết quả: {
  "entities": [
    {"name": "iPhone", "type": "PRODUCT"},
    {"name": "Apple", "type": "ORGANIZATION"},
    {"name": "2007", "type": "DATE"}
  ],
  "relationships": [
    {"source": "iPhone", "target": "Apple", "relation": "produced_by"},
    {"source": "iPhone", "target": "2007", "relation": "released_in"}
  ]
}

### Ví dụ 8:
Câu hỏi: "Albert Einstein đoạt giải Nobel Vật lý vào năm 1921."
Kết quả: {
  "entities": [
    {"name": "Albert Einstein", "type": "PERSON"},
    {"name": "Nobel Vật lý", "type": "AWARD"},
    {"name": "1921", "type": "DATE"}
  ],
  "relationships": [
    {"source": "Albert Einstein", "target": "Nobel Vật lý", "relation": "awarded"},
    {"source": "Albert Einstein", "target": "1921", "relation": "awarded_in"}
  ]
}

### Ví dụ 9:
Câu hỏi: "Facebook mua lại Instagram vào năm 2012 với giá 1 tỷ USD."
Kết quả: {
  "entities": [
    {"name": "Facebook", "type": "ORGANIZATION"},
    {"name": "Instagram", "type": "ORGANIZATION"},
    {"name": "2012", "type": "DATE"},
    {"name": "1 tỷ USD", "type": "MONEY"}
  ],
  "relationships": [
    {"source": "Facebook", "target": "Instagram", "relation": "acquired"},
    {"source": "Facebook", "target": "2012", "relation": "acquired_in"},
    {"source": "Facebook", "target": "1 tỷ USD", "relation": "acquired_for"}
  ]
}
### Ví dụ 10:
Câu hỏi: "Leonardo da Vinci là một họa sĩ nổi tiếng người Ý, tác giả của bức tranh Mona Lisa."
Kết quả: {
  "entities": [
    {"name": "Leonardo da Vinci", "type": "PERSON"},
    {"name": "Mona Lisa", "type": "ARTWORK"},
    {"name": "Ý", "type": "LOCATION"}
  ],
  "relationships": [
    {"source": "Leonardo da Vinci", "target": "Mona Lisa", "relation": "created"},
    {"source": "Leonardo da Vinci", "target": "Ý", "relation": "nationality"}
  ]
}
"""

driver = general.connect_to_graph_db()
model = general.load_model(model_name, system_instruction)
chunks = general.read_chunks(data_path)

rs = []
documents = []

for i in range(0, 27):
    start = i*10+i
    end = start + 10
    time.sleep(120)
    for chunk in range(start, end+1):
        try:
            content = chunks[chunk].page_content
            response = model.generate_content(content)
            extract = extract_relations_from_model_output(response.text)

            print(extract)
            if len(extract) != 0:
                rs.append(extract)
                create_nodes_and_relationships(driver, extract)
            print(chunk, ' thành công')
        except Exception as e:
            print(chunk, ' thất bại ', e)


# for r in rs:
#     create_nodes_and_relationships(driver, r)
driver.close()

# lấy tất cả quan hệ
# MATCH (n)-[r]->(m)
# RETURN n, r, m;
# xóa tất cả quan hệ
# MATCH (n)
# DETACH DELETE n;
