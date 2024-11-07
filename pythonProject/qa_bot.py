import os
from dotenv import load_dotenv
from qdrant_client import models

load_dotenv()
from langchain_core.prompts import PromptTemplate
import general
#

def matryoshka_prefetch(text_embedded_1024, text_embedded_768, text_embedded_512):
    return models.Prefetch(
        prefetch=[
            models.Prefetch(
                prefetch=[
                    models.Prefetch(
                        query=text_embedded_512,
                        using="matryoshka-512dim",
                        limit=100,
                    ),
                ],
                query=text_embedded_768,
                using="matryoshka-768dim",
                limit=50,
            )
        ],
        query=text_embedded_1024,
        using="default",
        limit=25,
    )

def query_from_db(client, collection_name, text_embedded_1024, text_embedded_768, text_embedded_512, embedded_late_interaction):
    return client.query_points(
        collection_name=f"{collection_name}",
        prefetch=[
            models.Prefetch(
                prefetch=[
                    models.Prefetch(
                        query=text_embedded_768,
                        using="matryoshka-768dim",
                        limit=100,
                    )
                ],
                query=text_embedded_1024,
                using="default",
                limit=25,
            ),
        ],
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,
        ),
    )

collection_name = os.getenv("name_collection")
model_name = os.getenv("model")

model_embedding_1024_name = os.getenv("model_embedding_1024")
model_embedding_768_name = os.getenv("model_embedding_768")
model_embedding_512_name = os.getenv("model_embedding_512")

model_splade_query_name = os.getenv("model_splade_query")
model_late_interaction_name = os.getenv("model_late_interaction")

url = os.getenv("url")

# 1. tải model
model = general.load_model(model_name)

# 2. tải model_embedding
model_1024, tokenizer1 = general.load_model_embedding(model_embedding_1024_name)
model_768, tokenizer2 = general.load_model_embedding(model_embedding_768_name)
model_512 = general.load_model_embedding_512(model_embedding_512_name)
model_late_interaction, tokenizer_late_interaction = general.load_late_interaction(model_late_interaction_name)

# 3. load model splade cho document
model_query, tokenizer_query = general.load_model_splade(model_splade_query_name)

# 4. tải database
client = general.load_db(url)

# 5. truy vấn database
query  = "hãy nói cho tôi về việc phân loại rag"

# 6. Tiền xử lý
vncorenlp = general.load_vncorenlp()
text_pre_processed = general.text_preprocessing_vietnamese(query.strip(), vncorenlp)

# 6. embedding query
text_embedded_1024 = general.get_embedding(model_1024, tokenizer1, text_pre_processed)
text_embedded_768 = general.get_embedding(model_768, tokenizer2, text_pre_processed)
text_embedded_512 = general.get_embedding_512(model_512, text_pre_processed)

# 7. embedding splade query
doc_vec, doc_tokens = general.compute_vector(text_pre_processed, model=model_query, tokenizer=tokenizer_query)
sorted_tokens_doc = general.extract_and_map_sparse_vector(doc_vec, tokenizer_query)

indices = tokenizer_query.convert_tokens_to_ids(sorted_tokens_doc)
values = list(sorted_tokens_doc.values())

embedded_late_interaction = general.get_embedding_late_interaction(model_late_interaction, tokenizer_late_interaction, text_pre_processed)

print("query:", text_pre_processed)

results = query_from_db(client, collection_name, text_embedded_1024, text_embedded_768, text_embedded_512, embedded_late_interaction)
print(results)

for result in results.points:
    print(result.payload["text"], "\n---------------------------------------------------------------------------------------------")























# scored_point_1  = query_from_db(client, name_collection, model_embedding, query)[0]
# scored_point_2  = query_from_db(client, name_collection, model_embedding, query)[1]
# scored_point_3  = query_from_db(client, name_collection, model_embedding, query)[2]
#
# page_content_1 = scored_point_1.payload['page_content']
# page_content_2 = scored_point_2.payload['page_content']
# page_content_3 = scored_point_3.payload['page_content']
#
# result_from_db = query + ", " + page_content_1 + ", " + page_content_2 + ", " + page_content_3
# print(query_from_db(client, name_collection, model_embedding, query))
#
# result_from_db = " ".join(result_from_db.split())
#
# template = """Bạn là một con chatbot ai dùng để trả lời câu hỏi, hãy trả lời chính xác. Câu hỏi của tôi là: {text}. Nếu không biết thì hãy nói không biết"""
# prompt = PromptTemplate(input_variables=["text"], template=template)
# formatted_prompt = prompt.format(text="cho toi 1 vai cau tho ve bien")
#
# response = model.generate_content(formatted_prompt)
# print(response.text)