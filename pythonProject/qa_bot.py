from dotenv import load_dotenv
import requests
load_dotenv()
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import general
import google.generativeai as genai
from qdrant_client import models
import os
from dotenv import load_dotenv
load_dotenv()


def load_model(model):
    genai.configure(api_key=os.getenv("API_KEY"))
    model = genai.GenerativeModel(model)
    print("load model success")
    return model

def query_from_db(client, collection_name, text_embedded_1024, text_embedded_768, text_embedded_512, embedded_late_interaction):
    return client.query_points(
        collection_name=f"{collection_name}",
        prefetch=models.Prefetch(
            prefetch=models.Prefetch(
                prefetch=models.Prefetch(
                    query=text_embedded_512,
                    using="matryoshka-512dim",
                    limit=100,
                ),
                query=text_embedded_768,
                using="matryoshka-768dim",
                limit=50,
            ),
            query=text_embedded_1024,
            using="matryoshka-1024dim",
            limit=25,
        ),
        query=embedded_late_interaction,
        using="late_interaction",
        limit=10,
    ).points

def re_ranking(query, query_text_json):
    invoke_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerankqa-mistral-4b-v3/reranking"

    headers = {
        "Authorization": f"Bearer {os.getenv("API_KEY_RERANKING")}",
        "Accept": "application/json",
    }

    payload = {
        "model": "nvidia/nv-rerankqa-mistral-4b-v3",
        "query": {
            "text": query
        },
        "passages": query_text_json
    }

    # re-use connections
    session = requests.Session()

    response = session.post(invoke_url, headers=headers, json=payload)

    response.raise_for_status()
    response_body = response.json()
    return response_body

def rewrite_query(model, query):
    system_rewrite = """
    ("system", "Bạn là một trợ lý hữu ích, tạo ra nhiều truy vấn tìm kiếm dựa trên một truy vấn đầu vào duy nhất.
    Thực hiện mở rộng truy vấn. Nếu có nhiều cách phổ biến để diễn đạt một câu hỏi của người dùng hoặc 
    có các từ đồng nghĩa phổ biến cho các từ khóa trong câu hỏi, hãy đảm bảo trả về nhiều phiên bản của truy vấn với các cách diễn đạt khác nhau.
    Nếu có các từ viết tắt hoặc từ bạn không quen thuộc, đừng cố gắng diễn đạt lại chúng.
    Trả về 3 phiên bản khác nhau của câu hỏi.")
    
    ("human", {question})
    """

    prompt = PromptTemplate(input_variables=["question"], template=system_rewrite)
    formatted_prompt = prompt.format(question=query)

    response = model.generate_content(formatted_prompt)

    return response.text

collection_name = os.getenv("name_collection")
model_name = os.getenv("model")

model_embedding_1024_name = os.getenv("model_embedding_1024")
model_embedding_768_name = os.getenv("model_embedding_768")
model_embedding_512_name = os.getenv("model_embedding_512")

model_splade_query_name = os.getenv("model_splade_query")
model_late_interaction_name = os.getenv("model_late_interaction")

url = os.getenv("url")

# 1. tải model
model = load_model(model_name)

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
query  = "Thực phẩm bao gói sẵn là gi?"

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

# 8. embedding late interaction query
embedded_late_interaction = general.get_embedding_late_interaction(model_late_interaction, tokenizer_late_interaction, text_pre_processed)

# 9. query từ database
print("query:", text_pre_processed)

results = query_from_db(client, collection_name, text_embedded_1024, text_embedded_768, text_embedded_512, embedded_late_interaction)
print(results)

json_query_text = []
print("\n----------------------------------------------------------------------------------------------------------------------------------------------------------")
for result in results:
    json_query_text.append({"text": result.payload["text"]})
    print(result.payload["text"], "\n-----------------------------------------------------------------------------------------------------------")


# 10. re-ranking
re_ranking_query_text = re_ranking(query, json_query_text)
print(re_ranking_query_text)