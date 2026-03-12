from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


loader = TextLoader('../Files/cricket.txt', encoding='utf-8')
text = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 50,  # no. of characters
    chunk_overlap = 0
)

res = splitter.split_text(text[0].page_content)

print(res)
print(len(res))
