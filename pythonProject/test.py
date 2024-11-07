
from transformers import AutoTokenizer, AutoModel

# Load the tokenizer and model from HuggingFace
tokenizer = AutoTokenizer.from_pretrained('colbert-ir/colbertv2.0')
model = AutoModel.from_pretrained('colbert-ir/colbertv2.0')

# Example query and document
query = """AI VIET NAM – COURSE 2023 Foundation of Prompt Engineering Ngày 19 tháng 3 năm 2024 Phần I: Tổng quan vềRAG Phần II: Retrieval Augmented Generation (RAG) Trong bối cảnh các mô hình ngôn ngữlớn (LLM) phát triển mạnh mẽ, sựxuất hiện của các mô hình GPT (OpenAI), LLama (Meta), Gemini (Google) đã thểhiện khảnăng ấn tượng trong việc sinh ngôn ngữ, thực hiện tác tác vụvới ngôn ngữtựnhiên. Cho dù vậy, các mô hình ngôn ngữlớn vẫn cho thấy còn nhiều điểm yếu như dữliệu thiếu tính cập nhật, thiếu dữliệu chuyên môn cho các lĩnh vực cụthể hay sinh ngôn ngữthiếu chính xác (hay được biết đến với thuật ngữ"hallucination")."""

query_tokens = tokenizer(query, return_tensors='pt', truncation=True, padding=True)

query_embedding = model(**query_tokens).last_hidden_state.detach().numpy()[0]

print(query_embedding)
print(len(query_embedding))