from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
from qdrant_client import QdrantClient, models
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import py_vncorenlp
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
from qdrant_client import models
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, NLTKTextSplitter, SpacyTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader, PDFMinerLoader

def load_model(model, system_instruction):
    genai.configure(api_key=os.getenv("API_KEY"))
    model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_instruction)

    print("load model success")
    return model

# đọc từng đoạn một của tất cả pdf
def read_chunks(data_path, chunk_size = 700, chunk_overlap = 140):
    # Khai báo loader để quét toàn bộ thư mục data
    loader = DirectoryLoader(data_path, glob="*.pdf", use_multithreading=True, loader_cls=PyMuPDFLoader)
    documents = loader.load()

    # Chỉ lấy từ trang 4 trở đi vì trang đầu thường là bìa sách và mục lục
    # documents = documents[4:len(documents)-2]

    # Sử dụng TextSplitter để chia nhỏ văn bản
    text_splitter = SpacyTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    print("Read chunk success")
    return chunks

def connect_to_graph_db():
    try:
        # Kết nối đến Neo4j
        uri = "bolt://localhost:7687"  # Địa chỉ Neo4j
        username = "neo4j"  # Tên đăng nhập
        password = "123456789"  # Mật khẩu
        driver = GraphDatabase.driver(uri, auth=(username, password))
        print("connect to graph db success")
        return driver
    except:
        print("connect to graph db failed")

def load_vncorenlp():
    return py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=r'C:\Users\Nam\Desktop\vncorenlp')

def text_preprocessing_vietnamese(text, vncorenlp):
    # 1. Chuyển thành chữ thường
    text = text.lower()

    # 2. Loại bỏ dấu câu
    text = re.sub(r'[^\w\s]', '', text)

    # 4. Xóa khoảng trắng thừa
    text = text.strip()

    # 5. Tách từ (Tokenization)
    output = vncorenlp.word_segment(text)
    result_string = ' '.join(output)

    return result_string

# kết nối đến db
def load_db(url):
    client = QdrantClient(url=url)
    print('load database success')
    return client

def create_collection(client, collection_name, size, distance):
    if client.collection_exists(collection_name=collection_name):
        return

    hnsw_config = {
        "m": 16,  # Số lượng liên kết trong HNSW
        "ef_construct": 1000,  # Số lượng vector cho việc xây dựng
        "full_scan_threshold": 10000  # Ngưỡng quét toàn bộ
    }

    # Tạo collection với cấu hình vectors và HNSW
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            'matryoshka-1024dim': {
                'size': size,
                'distance': distance,
                'datatype': models.Datatype.FLOAT32,
            },
            'matryoshka-768dim': {
                'size': 768,
                'distance': distance,
                'datatype': models.Datatype.FLOAT32,
            },
            'matryoshka-512dim': {
                'size': 512,
                'distance': distance,
                'datatype': models.Datatype.FLOAT32,
            },
            'late_interaction':{
                'size': 768,
                'distance': distance,
                'multivector_config': models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                'datatype': models.Datatype.FLOAT32
            }
        },
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            ),
        ),
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        },
        hnsw_config=hnsw_config  # Thêm cấu hình HNSW
    )

    print("create collection success")
    return client

def load_model_embedding(model_embedding_name):
    phobert = AutoModel.from_pretrained(model_embedding_name)
    tokenizer = AutoTokenizer.from_pretrained(model_embedding_name)
    print("load model embedding success")
    return phobert, tokenizer

def get_embedding(model_embedding, tokenizer, text_pre_processed):
    input_ids = torch.tensor([tokenizer.encode(text_pre_processed)])

    with torch.no_grad():
        features = model_embedding(input_ids)  # Models outputs are now tuples
        embeddings = features.last_hidden_state

    cls_embedding = embeddings[:, 0, :]
    return cls_embedding.numpy()[0]

# ------------------------------------------------------------------------------------------- #

def load_model_embedding_512(model_embedding_name):
    model = SentenceTransformer(model_embedding_name)
    print("load model embedding success")
    return model

def get_embedding_512(model_embedding, text_pre_processed):
    return model_embedding.encode(text_pre_processed)


def load_model_splade(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    print("load model embedding splade success")
    return model, tokenizer

def load_late_interaction(model_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("load model embedding late interaction success")
    return model, tokenizer

def get_embedding_late_interaction(model, tokenizer, text_pre_processed):
    query_tokens = tokenizer(text_pre_processed, return_tensors='pt', truncation=True, padding=True)

    query_embedding = model(**query_tokens).last_hidden_state.detach().numpy()[0]

    return query_embedding[: 10]

def compute_vector(text, model, tokenizer):
    tokens = tokenizer(text, return_tensors="pt")
    output = model(**tokens)
    logits, attention_mask = output.logits, tokens.attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    max_val, _ = torch.max(weighted_log, dim=1)
    vec = max_val.squeeze()

    return vec, tokens

def extract_and_map_sparse_vector(vector, tokenizer):

    # Extract indices and values of non-zero elements in the vector
    cols = vector.nonzero().squeeze().cpu().tolist()
    weights = vector[cols].cpu().tolist()

    # Map indices to tokens and create a dictionary
    idx2token = {idx: token for token, idx in tokenizer.get_vocab().items()}
    token_weight_dict = {idx2token[idx]: round(weight, 2) for idx, weight in zip(cols, weights)}

    # Sort the dictionary by weights in descending order
    sorted_token_weight_dict = {k: v for k, v in sorted(token_weight_dict.items(), key=lambda item: item[1], reverse=True)}

    return sorted_token_weight_dict
