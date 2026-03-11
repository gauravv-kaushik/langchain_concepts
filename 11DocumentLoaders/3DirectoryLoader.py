from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='CVs',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

# docs = loader.load()


print(docs)