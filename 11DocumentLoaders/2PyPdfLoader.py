from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader, UnstructuredPDFLoader, AmazonTextractPDFLoader, PyMuPDFLoader


loader = PyPDFLoader('../Files/GauravPythonDev_FTE.pdf')

docs = loader.load()

print(docs)

# Some Types of PDF Loaders which can work in different Scenerios

# 1. Simple, Clean PDF, text Data = PyPDFLoader
# 2. PDFs with table and columns = PDFPlumberLoader
# 3. Scanned Images PDF = UnstructuredPDFLoader or AmazonTextractPDFLoader
# 4. Need layout and Image Data = PyMuPDFLoader
