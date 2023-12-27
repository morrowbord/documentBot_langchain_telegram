# Overall Functionality:
# The code create a Telegram bot that loads documents into a vector store,
# allows for similarity searches against those documents, 
# and integrates with a question-answering chain using a Llama language model to provide responses.

import logging
import os
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, ContextTypes, CommandHandler
from langchain.document_loaders import TextLoader
#from langchain.text_splitter import CharacterTextSplitter
#from langchain.vectorstores.faiss import FAISS
#from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
#from os.path import join, dirname

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp


#logs
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
    )

load_dotenv()  # Load environment variables from .env file
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # Get the value of the telegram variable

vectorstore = None # #global

#старт бота
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, 
    text="Привет, Я тестовый бот")
    

async def load(update: Update, context: ContextTypes.DEFAULT_TYPE):
    loader = TextLoader('state_of_the_union.txt')
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    all_splits = text_splitter.split_documents(data)

    
    global vectorstore
    if vectorstore != None:
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=GPT4AllEmbeddings())
        await context.bot.send_message(chat_id=update.effective_chat.id,
        text='document alredy exist')
    else:
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings(), persist_directory="./chroma_db" )
        await context.bot.send_message(chat_id=update.effective_chat.id,
        text='document loaded')


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


async def query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    docs = vectorstore.similarity_search(update.message.text)
    chain = load_qa_chain(
    llm = LlamaCpp(
    model_path="[llama-2-7b-chat.Q4_K_S.gguf]",# change path to local model in [];
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
    ))

    results = chain({'input_documents': docs, 'question': update.message.text},return_only_outputs=True)
    text = results['output_text']

    await context.bot.send_message(chat_id=update.effective_chat.id, 
    text=text)


if __name__== "__main__":
    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler('start',start))
    application.add_handler(CommandHandler('ekofarm',load))
    application.add_handler(CommandHandler('query',query))


    application.run_polling()
