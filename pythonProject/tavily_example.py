from langchain_community.tools import TavilySearchResults
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("API_KEY_TAVILA")
tool = TavilySearchResults(
    max_results=5,
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

query = "trung quốc chip ai"
results = tool.invoke(query)
print(results)

