# from langchain_nvidia_ai_endpoints import NVIDIARerank
# from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
#
# reranker = NVIDIARerank()
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=reranker, base_retriever=retriever
# )
#
# reranked_chunks = compression_retriever.compress_documents(query)
import os
import general
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import FakeEmbeddings

model_embedding_512_name = os.getenv("model_embedding_512")
model_512 = general.load_model_embedding_512(model_embedding_512_name)
text = """
Học sâu (Deep Learning) là một lĩnh vực con của học máy (Machine Learning) tập trung vào việc sử dụng các mạng nơ-ron nhân tạo nhiều lớp để học từ dữ liệu lớn. Các ứng dụng của học sâu rất đa dạng, từ nhận diện giọng nói, hình ảnh, đến xử lý ngôn ngữ tự nhiên. 
Trong khi học sâu phát triển mạnh, học máy truyền thống vẫn có nhiều ứng dụng quan trọng, đặc biệt là trong các bài toán nhỏ gọn, dữ liệu hạn chế.
"""

text_splitter = SemanticChunker(
    {'embed_documents': general.get_embedding_512(model_512, text)}, breakpoint_threshold_type="percentile"
)

text = """
Học sâu (Deep Learning) là một lĩnh vực con của học máy (Machine Learning) tập trung vào việc sử dụng các mạng nơ-ron nhân tạo nhiều lớp để học từ dữ liệu lớn. Các ứng dụng của học sâu rất đa dạng, từ nhận diện giọng nói, hình ảnh, đến xử lý ngôn ngữ tự nhiên. 
Trong khi học sâu phát triển mạnh, học máy truyền thống vẫn có nhiều ứng dụng quan trọng, đặc biệt là trong các bài toán nhỏ gọn, dữ liệu hạn chế.
"""
docs = text_splitter.create_documents([text])
print(docs)
