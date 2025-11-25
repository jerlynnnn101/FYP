import os
import sys
from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA 
from langchain_ibm import WatsonxLLM

# --- 1. IBM Watsonx.ai Credentials and Connection Setup (No Change) ---
os.environ["WATSONX_API_KEY"] = "heqOEuyYXy-ngbBIHva-GR8-0HNBFystyN9V6Vv2oMJB"
os.environ["IBM_PROJECT_ID"] = "0ed5949a-bd21-4f66-9733-4646506adc34"
os.environ["WATSONX_URL"] = "https://us-south.ml.cloud.ibm.com" 

if not os.getenv("WATSONX_API_KEY"):
    print("FATAL ERROR: WATSONX_API_KEY is missing. Cannot proceed.")
    sys.exit(1)

print("✅ IBM Credentials and URL loaded.")

# --- 2. RAG Application Setup (Vector Store remains the same) ---

watsonx_llm = WatsonxLLM(                      
    model_id="ibm/granite-3-8b-instruct",                   
    project_id=os.getenv("IBM_PROJECT_ID"),
    params={'max_new_tokens': 1024, 'temperature': 0.1} 
)
print("✅ WatsonxLLM initialized.")

directory_path = './granite-snack-cookbook/fyp_document'
loader = DirectoryLoader(
    path=directory_path, 
    loader_cls=PDFPlumberLoader, 
    glob='*.pdf'             
)

try:
    documents = loader.load()
except Exception as e:
    print(f"FATAL ERROR: Failed to load documents from {directory_path}. Error: {e}")
    sys.exit(1)

text_splitter = RecursiveCharacterTextSplitter( 
    chunk_size=300,        
    chunk_overlap=150     
)
docs = text_splitter.split_documents(documents)
print(f"✅ Documents split into {len(docs)} chunks.")

# The HuggingFaceEmbeddings model is powerful, but we need to tune retrieval.
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2" 
)

db = FAISS.from_documents(docs, embeddings) 
print("✅ FAISS Vector Index created.")

# --- 3. Create the RetrievalQA Chain (CRITICAL TWEAK HERE) ---
# We are changing the search type and reducing the number of documents passed to the LLM.

qa_chain = RetrievalQA.from_chain_type(
    llm=watsonx_llm,        
    chain_type="stuff",        
    # 🌟 CRITICAL TWEAK: Set search_type to 'mmr' and k=3 to prioritize the most relevant unique chunks.
    retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 3}), 
    return_source_documents=True 
)       

print("--- OPTIMIZED RAG Setup Complete. Ready to query. ---")

# --- 4. Your Original Question Cell (Execution) ---

if qa_chain:
    question = "What are the ISBN and DOI for the paperback version of the STEM Education report?"

    print(f"\n--- Running Query: {question} ---")

    try:
        result_dict = qa_chain.invoke({'query': question})

        answer = result_dict['result']
        sources = result_dict['source_documents']

        print("\n--------------------------------------------------------------------------------")
        print(f"Question: {question}")
        print("---")
        print(f"Answer: {answer}")

        # --- CITATION SECTION ---
        print("\n--- Sources Used (for Citation) ---")
        for doc in sources:
            source_file = doc.metadata.get('source', 'N/A').split('/')[-1]
            page_number = doc.metadata.get('page', 'N/A')
            print(f"File: {source_file} (Page: {page_number})")
            
    except Exception as e:
        print(f"\nQUERY FAILED during execution. Error details: {e}")

else:
    print("\nQuery cannot be executed because the RAG chain failed to initialize.")