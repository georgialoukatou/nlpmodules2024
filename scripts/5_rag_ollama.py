#import os
#from langchain_community.llms import Ollama
#from dotenv import load_dotenv
#from langchain.document_loaders import TextLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.vectorstores import Chroma
#from langchain.chains import create_retrieval_chain
#from langchain import hub
#from langchain.chains.combine_documents import create_stuff_documents_chain
"""
import json
import openai

from openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS



list_descriptions = ["L'iPhone 14 Pro combine un écran Super Retina XDR de 6,1 pouces. ", "Le Samsung Galaxy S23 Ultra dispose d'un écran AMOLED de 6,8 pouces"]

hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Diviser les documents en chunks (si nécessaire)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
chunks = [text_splitter.split_text(description) for description in list_descriptions]

print(chunks)


vectorstore = FAISS.from_texts(
    texts=list_descriptions, #[doc["content"] for doc in documents],
    embedding=hf_embeddings,
    #metadatas=[doc["metadata"] for doc in documents]
)

retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# Créez une chaîne de récupération QA
#qa_chain = RetrievalQA.from_chain_type(
#    llm=llm,
#    retriever=retriever,
#    return_source_documents=True  # Renvoie les documents sources
#)

# Exemple de requête utilisateur
query = "Quelle taille ecran pour Samsung Galaxy ?"
#response = qa_chain({"query": query})

#print("Answer:", response["result"])
#print("Sources:", response["source_documents"])

#deuxieme
retrieved_docs = retriever.get_relevant_documents(query)
print(retrieved_docs)
context = "\n\n".join([doc.page_content for doc in retrieved_docs])
"""
# Créer un prompt personnalisé
#prompt = f"""
#Vous êtes un assistant intelligent qui répond aux questions en utilisant uniquement les informations ci-dessous :
#{context}

#Question : {query}
#Répondez de manière concise et précise.
#"""
#response = llm.predict(prompt)
#print("Réponse :", response)
#"""

# Créer le contexte à partir des documents récupérés
#context = "\n\n".join([doc.page_content for doc in retrieved_docs])

# Créer un prompt personnalisé
#prompt = f"""
#Vous êtes un assistant intelligent qui répond aux questions en utilisant uniquement les informations ci-dessous :
#{context}

#Question : Qu'est-ce que ChromaDB ?
#Répondez de manière concise et précise.


#from langchain.chat_models import ChatOpenAI

# Initialisez le modèle LLM
#llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key="OPENAI_KEY")

# Obtenez la réponse directement
#response = llm.predict(prompt)

#print("Réponse :", response)




"""
"""
"""
client = OpenAI(
    api_key=OPENAI_KEY,  # This is the default and can be omitted
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
)

print("chat_competion " + str(chat_completion))
"""
"""
"""
# Load the JSONL data properly
docs = []
#with open("filtered_meta_1000.jsonl", "r", encoding="utf-8") as f:
#    for line in f:
#        # Parse each JSON line into a Document with the 'title' and 'description'
#        data = json.loads(line)
#        combined_text = f"Title: {data.get('title', '')}\nDescription: {' '.join(data.get('description', []))}"
#        docs.append(combined_text)


#print(str(docs))

#llm = Ollama(model="llama3.1", base_url="http://127.0.0.1:11434")
#


from langchain import hub
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

list_descriptions = [
    "L'iPhone 14 Pro combine un écran Super Retina XDR de 6,1 pouces.",
    "Le Samsung Galaxy S23 Ultra dispose d'un écran AMOLED de 6,8 pouces."
]
llm = OllamaLLM(model="llama3.1", base_url="http://127.0.0.1:11434")

embed_model = OllamaEmbeddings(
    model="llama3.1",
    base_url='http://127.0.0.1:11434'
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
chunks = text_splitter.split_text(str(list_descriptions))
vector_store = Chroma.from_texts(chunks, embed_model)

retriever = vector_store.as_retriever()
chain = create_retrieval_chain(combine_docs_chain=llm,retriever=retriever)
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)    

response = retrieval_chain.invoke({"input": "Dis moi sur iphone14"})
print("################################")
print(response['answer'])
