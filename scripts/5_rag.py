import openai
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
load_dotenv()

# === Configuration ===
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# === Étape 1 : Préparer les descriptions ===
list_descriptions = [
    "L'iPhone 14 Pro combine un écran Super Retina XDR de 6,1 pouces.",
    "Le Samsung Galaxy S23 Ultra dispose d'un écran AMOLED de 6,8 pouces."
]

# === Étape 2 : Initialiser les embeddings ===
hf_embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# === Étape 3 : Créer la base vectorielle avec FAISS ===
vectorstore = FAISS.from_texts(
    texts=list_descriptions,
    embedding=hf_embeddings
)

retriever = vectorstore.as_retriever()

# === Étape 4 : Configurer le modèle LLM ===
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# === Étape 5 : Interroger le système ===
def ask_question(query, retriever, llm):
    """
    Fonction pour poser une question à la base vectorielle
    et obtenir une réponse du LLM avec un prompt personnalisé.
    """
    # Récupérer les documents pertinents
    retrieved_docs = retriever.get_relevant_documents(query)
    
    # Construire le contexte avec les documents récupérés
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Créer le prompt personnalisé
    prompt = f"""
    Vous êtes un assistant intelligent qui répond aux questions en utilisant uniquement les informations ci-dessous :
    {context}

    Question : {query}
    Répondez de manière concise et précise.
    """
    # Obtenir la réponse du LLM
    response = llm.predict(prompt)
    return response, retrieved_docs

# === Étape 6 : Exemple d'utilisation ===
query = "Quelle taille d'écran pour Samsung Galaxy ?"

#response, sources = ask_question(query, retriever, llm)

# === Résultats ===
#print("Réponse avec FAISS :", response)


######################## SI ON VEUT UTILISER CHROMA ###########################################

from langchain.vectorstores import Chroma
CHROMA_DB_DIR = "./chroma_db"  # Répertoire pour stocker la base Chroma

vectorstore = Chroma.from_texts(
    texts=list_descriptions,
    embedding=hf_embeddings,
    persist_directory=CHROMA_DB_DIR  # Permet de sauvegarder la base
)

retriever = vectorstore.as_retriever()


#response, sources = ask_question(query, retriever, llm)

# === Résultats ===
#print("Réponse avec Chroma :", response)


######################## SI ON VEUT UTILISER HUGGINGFACE ###########################################



from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# === Étape 1 : Préparer les descriptions ===
list_descriptions = [
    "L'iPhone 14 Pro combine un écran Super Retina XDR de 6,1 pouces.",
    "Le Samsung Galaxy S23 Ultra dispose d'un écran AMOLED de 6,8 pouces."
]

# === Étape 2 : Initialiser les embeddings ===
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
hf_embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# === Étape 3 : Créer la base vectorielle avec FAISS ===
vectorstore = FAISS.from_texts(
    texts=list_descriptions,
    embedding=hf_embeddings
)

retriever = vectorstore.as_retriever()

# === Étape 4 : Configurer le modèle LLM ===
# Charger le modèle Hugging Face pour la génération
HF_LLM_MODEL = "bigscience/bloomz-560m"  # Exemple de modèle
tokenizer = AutoTokenizer.from_pretrained(HF_LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(HF_LLM_MODEL)

# === Étape 5 : Interroger le système ===
def ask_question_hf(query, retriever, model, tokenizer):
    """
    Fonction pour poser une question à la base vectorielle
    et obtenir une réponse du LLM Hugging Face avec un prompt personnalisé.
    """
    # Récupérer les documents pertinents
    retrieved_docs = retriever.get_relevant_documents(query)
    
    # Construire le contexte avec les documents récupérés
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    query = "Tell me about OnePlus 6T"
    context = "a resilient glass back, the OnePlus 6T was crafted with care and purpose. Experience a 6.41 inch Optic AMOLED display for true immersion through an 86% screen-to-body ratio, beautifully slim cut-out, and more. Nail every detail night or day with Opticalwith up to 8GB of RAM, the Qualcomm Snapdragon 845 Mobile Platform. Hardware and software work together for an experience that is consistently Fast and Smooth. With a 3700 mAh capacity, the OnePlus 6T puts more power into your hands than ever, while ourinto your hands than ever, while our Fast Charge technology gets you up and running in just half an hour. Our operating system is all about ensuring your phone works for you, not the other way around. Powered by Android Pie, the OnePlus 6T offers moreoffer superior sound quality and convenience. Learn why the Type-C Bullets are the perfect wired companion to your OnePlus 6T."
    # Créer le prompt personnalisé
    prompt = f"""
    Vous êtes un assistant intelligent qui répond aux questions en utilisant uniquement les informations ci-dessous :
    {context}

    Question : {query}
    """
    
    # Préparer les entrées pour le modèle
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False, max_length = 512)
    
    # Générer une réponse
    outputs = model.generate(inputs["input_ids"], max_new_tokens = 500, max_length=512)
    
    # Décoder la réponse
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response, retrieved_docs

# === Étape 6 : Exemple d'utilisation ===
query = "Quelle taille d'écran pour Samsung Galaxy ?"

response, sources = ask_question_hf(query, retriever, model, tokenizer)

# === Résultats ===
print("Réponse avec Hugging Face :", response)