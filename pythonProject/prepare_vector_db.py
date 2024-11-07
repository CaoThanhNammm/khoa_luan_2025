from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, NLTKTextSplitter, SpacyTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
import os
from qdrant_client import models
from dotenv import load_dotenv
from qdrant_client.models import PointStruct
load_dotenv()
import general
import random

# đọc từng đoạn một của tất cả pdf
def read_chunks(data_path, chunk_size = 700, chunk_overlap = 140):
    # Khai báo loader để quét toàn bộ thư mục data
    loader = DirectoryLoader(data_path, glob="*.pdf", use_multithreading=True, loader_cls=PyMuPDFLoader)
    documents = loader.load()

    # Chỉ lấy từ trang 4 trở đi vì trang đầu thường là bìa sách và mục lục
    # documents = documents[4:]

    # Sử dụng TextSplitter để chia nhỏ văn bản
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    print("Read chunk success")
    return chunks

# tạo embedding với len(embedding) = 768
def create_embedding(chunks, collection_name, pho_bert_large, pho_bert_base_v2, model_512, tokenizer1, tokenizer2, model_splade_doc, tokenizer_splade_doc, model_late_interaction, tokenizer_late_interaction, client):
    vncorenlp = general.load_vncorenlp()

    points_1024 = []
    points_768 = []
    points_512 = []
    points_sparse_vector = []
    points_late_interaction = []
    id = 1

    for chunk in range(len(chunks)):
        try:
            metadata = chunks[chunk].metadata
            content = chunks[chunk].page_content

            text_pre_processing = general.text_preprocessing_vietnamese(content, vncorenlp)

            embedded_text_1024 = general.get_embedding(pho_bert_large, tokenizer1, text_pre_processing)
            embedded_text_768 = general.get_embedding(pho_bert_base_v2, tokenizer2, text_pre_processing)
            embedded_text_512 = general.get_embedding_512(model_512, text_pre_processing)

            doc_vec, doc_tokens = general.compute_vector(text_pre_processing, model=model_splade_doc, tokenizer=tokenizer_splade_doc)
            sorted_tokens_doc = general.extract_and_map_sparse_vector(doc_vec, tokenizer_splade_doc)

            embedded_late_interaction = general.get_embedding_late_interaction(model_late_interaction, tokenizer_late_interaction, text_pre_processing)

            indices = tokenizer_splade_doc.convert_tokens_to_ids(sorted_tokens_doc)
            values = list(sorted_tokens_doc.values())

            print(text_pre_processing)

            points_1024.append(
                PointStruct(
                    id=id,
                    payload={"text": content, "metadata": metadata},
                    vector = {
                        'default': embedded_text_1024
                    }
                )
            )
            id += 1
            points_768.append(
                PointStruct(
                    id=id,
                    payload={"text": content, "metadata": metadata},
                    vector={
                        'matryoshka-768dim': embedded_text_768
                    }
                )
            )
            id += 1
            points_512.append(
                PointStruct(
                    id=id,
                    payload={"text": content, "metadata": metadata},
                    vector={
                        'matryoshka-512dim': embedded_text_512
                    }
                )
            )
            id += 1
            points_sparse_vector.append(
                PointStruct(
                    id=id,
                    payload={"text": content, "metadata": metadata},
                    vector={
                        'keyword': models.SparseVector(indices=indices, values=values)
                    }
                )
            )
            id += 1
            points_late_interaction.append(
                PointStruct(
                    id=id,
                    payload={"text": content, "metadata": metadata},
                    vector={
                        'late_interaction': embedded_late_interaction
                    }
                )
            )
            id += 1
        except Exception as e:
            print(e)
            continue

    client.upsert(collection_name, points_1024)
    client.upsert(collection_name, points_768)
    client.upsert(collection_name, points_512)
    client.upsert(collection_name, points_sparse_vector)
    client.upsert(collection_name, points_late_interaction)

    print("create embedding success")

model_name = os.getenv("model")
data_path = os.getenv("data_path")

model_embedding_1024_name = os.getenv("model_embedding_1024")
model_embedding_768_name = os.getenv("model_embedding_768")
model_embedding_512_name = os.getenv("model_embedding_512")

model_late_interaction_name = os.getenv("model_late_interaction")

url = os.getenv("url")
collection_name = os.getenv("name_collection")
size = os.getenv("size")
distance = os.getenv("distance")

model_splade_doc_name = os.getenv("model_splade_doc")

# 1. tải model
model = general.load_model(model_name)

# 2. load model và model_embedding để embedding
model_1024, tokenizer1 = general.load_model_embedding(model_embedding_1024_name)
model_768, tokenizer2 = general.load_model_embedding(model_embedding_768_name)
model_512 = general.load_model_embedding_512(model_embedding_512_name)

model_late_interaction, tokenizer_late_interaction = general.load_late_interaction(model_late_interaction_name)

# 3. load model splade cho document
model_doc, tokenizer_doc = general.load_model_splade(model_splade_doc_name)

# 4. tải database
client = general.load_db(url)

# 5. tạo collection, nếu có rồi thì không tạo nữa
collection = general.create_collection(client, collection_name, size, distance)

# 6. lấy ra chunks trong tất cả các doc
chunks = read_chunks(data_path)

# 7. tạo embedding
create_embedding(chunks, collection_name, model_1024, model_768, model_512, tokenizer1, tokenizer2, model_doc, tokenizer_doc, model_late_interaction, tokenizer_late_interaction, client)


