import os
import json
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


with open('current_context.txt', 'r') as file:
    current_context = file.read()

with open('system_prompt.txt', 'r') as file:
    system_prompt = file.read()

class GroqBackend:
    def __init__(self, api_key, model_name="llama-3.1-70b-versatile", docs_folder="./rag_files"):
        self.api_key = api_key
        self.model_name = model_name
        self.docs_folder = docs_folder
        self.setup_llm()
        self.setup_embeddings()
        self.setup_vectorstore()
        self.setup_memory()
        self.setup_mira_personality()
        self.setup_rag_chain()
        self.conversation_file = self.create_conversation_file()

    def setup_llm(self):
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name
        )
        print("--------------------------------------------------")
        print(f"LLM set up with model: {self.model_name}")
        print("--------------------------------------------------")

    def setup_embeddings(self):
        self.embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        print("--------------------------------------------------")
        print("Embedding model set up: BAAI/bge-small-en-v1.5")
        print("--------------------------------------------------")

    def setup_vectorstore(self):
        self.load_documents()
        self.vectorstore = Chroma.from_documents(self.documents, self.embed_model)
        self.retriever = self.vectorstore.as_retriever()
        print("--------------------------------------------------")
        print(f"Vectorstore set up with {len(self.documents)} documents")
        print("--------------------------------------------------")

    def load_documents(self):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.documents = []
        for filename in os.listdir(self.docs_folder):
            if filename.endswith(".txt") or filename.endswith(".md") or filename.endswith(".json"):
                with open(os.path.join(self.docs_folder, filename), 'r') as file:
                    text = file.read()
                    chunks = text_splitter.split_text(text)
                    for chunk in chunks:
                        self.documents.append(Document(page_content=chunk, metadata={"source": filename}))

        print("--------------------------------------------------")
        print(f"Loaded {len(self.documents)} document chunks from {self.docs_folder}")
        print("--------------------------------------------------")

    def setup_memory(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        print("--------------------------------------------------")
        print("Conversation memory initialized")
        print("--------------------------------------------------")

    def setup_mira_personality(self):
        self.mira_persona = f"""
        {system_prompt}

        Current context:
        - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        {current_context}
        """
        print("--------------------------------------------------")
        print("Mira's personality set up")
        print("--------------------------------------------------")

    def setup_rag_chain(self):
        rag_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                self.mira_persona + "\nUse the following pieces of context to inform your response, but don't explicitly mention them: {context}"
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def retrieve_and_format(query):
            docs = self.retriever.invoke(query)
            print("--------------------------------------------------")
            print(f"Retrieved {len(docs)} documents for query: {query}")
            print("--------------------------------------------------")
            for i, doc in enumerate(docs, 1):
                print(f"Document {i}:")
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"Content: {doc.page_content[:200]}...")  # Print first 200 characters
                print("--------------------------------------------------")
            formatted_docs = format_docs(docs)
            print(f"Formatted context (first 500 characters): {formatted_docs[:500]}...")
            print("--------------------------------------------------")
            return formatted_docs

        self.rag_chain = (
            {
                "context": RunnableLambda(retrieve_and_format),
                "human_input": RunnablePassthrough(),
                "chat_history": RunnableLambda(lambda x: self.memory.load_memory_variables({})["chat_history"]),
            }
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )
        print("--------------------------------------------------")
        print("RAG chain set up")
        print("--------------------------------------------------")

    def get_response(self, user_input):
        print("--------------------------------------------------")
        print(f"User input: {user_input}")
        print("--------------------------------------------------")
        
        response = self.rag_chain.invoke(user_input)
        
        print("--------------------------------------------------")
        print(f"Mira's response: {response}")
        print("--------------------------------------------------")
        
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(response)
        self.save_message("Human", user_input)
        self.save_message("Mira", response)
        return response

    def create_conversation_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
        file_path = os.path.join(self.docs_folder, filename)
        with open(file_path, 'w') as f:
            json.dump({"messages": []}, f)
        print("--------------------------------------------------")
        print(f"Conversation file created: {file_path}")
        print("--------------------------------------------------")
        return file_path

    def save_message(self, speaker, message):
        with open(self.conversation_file, 'r+') as f:
            data = json.load(f)
            data["messages"].append({"speaker": speaker, "message": message})
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
        # print("--------------------------------------------------")
        # print(f"Message saved: {speaker}: {message}")
        # print("--------------------------------------------------")

    def update_conversation_title(self, title):
        with open(self.conversation_file, 'r+') as f:
            data = json.load(f)
            data["title"] = title
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
        print("--------------------------------------------------")
        print(f"Conversation title updated: {title}")
        print("--------------------------------------------------")