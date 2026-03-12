from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader


loader = TextLoader('../Files/cricket.txt', encoding='utf-8')
text = loader.load()


splitter = CharacterTextSplitter(
    separator='',
    chunk_size = 50,
    chunk_overlap = 10   # It is common text that will be same in both chunks
)

res = splitter.split_text(text[0].page_content)

print(res)
print(len(res))


pdf_loader = PyPDFLoader('../Files/GauravPythonDev_FTE.pdf')

docs = pdf_loader.load()

pdf_res = splitter.split_documents(docs)


print(pdf_res)
print(len(pdf_res))