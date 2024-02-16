import logging

from chromadb.config import Settings
from huggingface_hub import snapshot_download
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import \
    RecursiveCharacterTextSplitter
from langchain_community.document_loaders import \
    PDFMinerLoader
from langchain_community.embeddings import \
    HuggingFaceInstructEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, pipeline, \
    AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
MODEL_DIR = r"C:\MODELS"

# Załadowanie pliku PDF
loader = PDFMinerLoader("war-and-peace.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Utworzenie bazy danych
embeddings_path = snapshot_download(
    repo_id="hkunlp/instructor-xl",
    cache_dir=MODEL_DIR,
    resume_download=True)
embeddings = HuggingFaceInstructEmbeddings(
    model_name=embeddings_path,
    model_kwargs={"device": "cuda"})

db = Chroma.from_documents(
    texts,
    embeddings,
    client_settings=Settings(anonymized_telemetry=False)
)
retriever = db.as_retriever()

# Załadowanie modelu oraz utworzenie potoku
tokenizer = AutoTokenizer.from_pretrained(
    "TheBloke/vicuna-13B-v1.5-GPTQ",
    cache_dir=MODEL_DIR,
    device_map="auto")
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/vicuna-13B-v1.5-GPTQ",
    cache_dir=MODEL_DIR,
    device_map="auto")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=4096,
)
local_llm = HuggingFacePipeline(pipeline=pipe)

# Utworzenie łańcucha zapytania
prompt_template = '''A chat between a curious user and an 
artificial intelligence assistant. The assistant gives 
helpful and detailed answers to the user's questions. 
Context: {context}

USER: {question} 

ASSISTANT:'''

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template)
qa = RetrievalQA.from_chain_type(llm=local_llm,
                                 chain_type="stuff",
                                 retriever=db.as_retriever(),
                                 return_source_documents=True,
                                 chain_type_kwargs={
                                     "prompt": prompt})

# Wysłanie pytania do modelu
query = "Tell me the story of Pierre."
res = qa(query)

# Wyświetlenie odpowiedzi oraz źródła
answer, docs = res['result'], res['source_documents']

print(answer)
print("\n---\n".join(
    [f"Source {i + 1}:\n{document.page_content}" for
     i, document in enumerate(docs)]))
