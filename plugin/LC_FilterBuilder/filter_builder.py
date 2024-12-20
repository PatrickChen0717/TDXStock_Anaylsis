from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
import langchain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationKGMemory
from langchain.prompts.prompt import PromptTemplate

import pandas as pd
import openai
import os
import langchain
from dotenv import load_dotenv
load_dotenv('plugin/LC_FilterBuilder/.env')

openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_type = os.getenv('OPENAI_API_TYPE')
openai.api_version = os.getenv('OPENAI_API_VERSION')

filter_dir = "plugin"

def read_existing_files():
    file_list = []
    for filename in os.listdir(filter_dir):
        filepath = os.path.join(filter_dir, filename)
        if os.path.isdir(filepath):
            continue

        filepath = os.path.join(filter_dir, filename)
        print(f"Reading: {filepath}")
        file_list.append(filepath)

    return file_list

# read_existing_files()

class filter_builder():
    memory = {}

    def __init__(self, model_name = "gpt-35-turbo-16k", docs = []):
        self.model = langchain.chat_models.AzureChatOpenAI(deployment_name=model_name,
                                            model_name=model_name, 
                                            temperature=0,
                                            verbose=True)
        # raw_text = self.get_text(docs)
        with open("C:\\Users\\I590127\\Documents\\identity.txt", 'r', encoding='utf-8') as file:
            raw_text = file.read()
        text_chunks = self.get_text_chunks(raw_text)
        vectorstore = self.get_vectorstore(text_chunks)

        self.conversation = self.get_conversation_chain(vectorstore)

    def get_text(self, docs):
        text = ""
        for doc in docs:
            f = open(doc, 'r')
            content = f.read()
            f.close()
            text += content
        return text

    def get_text_chunks(self, raw_text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(raw_text)
        return chunks

    def get_vectorstore(self,text_chunks):
        embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
        vectorstore = FAISS.from_texts(texts = text_chunks, 
                                    embedding = embeddings)
        return vectorstore
    
    def get_conversation_chain(self,vectorstore):
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages = True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm = self.model,
            retriever=vectorstore.as_retriever(),
            verbose=True, 
            memory = memory,
        )

        return conversation_chain
    
    def input(self, input_prompt):
        response = self.conversation.run({'question':input_prompt})

        print(response)


FB = filter_builder(docs = read_existing_files())
# template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. 
#             If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

#             Relevant Information:

#             {history}

#             Conversation:
#             Human: {input}
#             AI:"""
# prompt = PromptTemplate(input_variables=["history", "input"], template=template)

# llm = FB.model
# conversation_with_kg = ConversationChain(
#     llm=llm, verbose=True, prompt=prompt, memory=ConversationKGMemory(llm=llm)
# )

while True:
    user_input = input("Input >>>")
    
    if user_input == "exit":
        break 
    else:
        print(FB.input(user_input))



# while True:
#     response = ""
#     user_input = input("Input >>>")

#     if user_input == "exit":
#         break 
#     else:
#         print(conversation_with_kg.predict(input=user_input))


'''
self-knowledge : “Give an introduction of yourself” or “Describe your typical weekday schedule in broad strokes”
memory (short-term mem) : “Who is [name]?” or “Who is running for mayor?
plans (long-term mem) :  “What will you be doing at 10 am tomorrow?”
'''