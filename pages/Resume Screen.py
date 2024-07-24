
# langchain: https://python.langchain.com/
from dataclasses import dataclass
import streamlit as st
from speech_recognition.openai_whisper import save_wav_file, transcribe
from audio_recorder_streamlit import audio_recorder
from langchain_community.callbacks.manager import get_openai_callback
# from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationChain
from langchain.prompts.prompt import PromptTemplate
from prompts.prompts import templates
from typing import Literal
# from aws.synthesize_speech import synthesize_speech
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import NLTKTextSplitter
from PyPDF2 import PdfReader
from prompts.prompt_selector import prompt_sector
from streamlit_lottie import st_lottie
import json
# from IPython.display import Audio
import nltk

from pages.textgen import TextGen
hyperparams = {
    'temperature': 0.3,
    'top_p': 0.9,
    'top_k': 50,
    'max_tokens': 150,
    'repetition_penalty': 1.2
}

model_url = 'https://beside-arrested-queensland-drum.trycloudflare.com'

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
st_lottie(load_lottiefile("images/welcome.json"), speed=1, reverse=False, loop=True, quality="high", height=300)

with st.expander("""Why did I encounter errors when I tried to talk to the AI Interviewer?"""):
    st.write("""This is because the app failed to record. Make sure that your microphone is connected and that you have given permission to the browser to access your microphone.""")
with st.expander("""Why did I encounter errors when I tried to upload my resume?"""):
    st.write("""
    Please make sure your resume is in pdf format. More formats will be supported in the future.
    """)

st.markdown("""\n""")
position = st.selectbox("Select the position you are applying for", ["Data Scientist", "Software Engineer", "Marketing"])
resume = st.file_uploader("Upload your resume", type=["pdf"])
auto_play = st.checkbox("Let AI interviewer speak! (Please don't switch during the interview)")

@dataclass
class Message:
    """Class for keeping track of interview history."""
    origin: Literal["human", "ai"]
    message: str

def save_vector(resume):
    """embeddings"""
    nltk.download('punkt')
    pdf_reader = PdfReader(resume)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    # Split the document into chunks
    text_splitter = NLTKTextSplitter()
    texts = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    docsearch = FAISS.from_texts(texts, embeddings)
    return docsearch

def initialize_session_state_resume():
    # convert resume to embeddings
    if 'docsearch' not in st.session_state:
        st.session_state.docserch = save_vector(resume)
    # retriever for resume screen
    if 'retriever' not in st.session_state:
        st.session_state.retriever = st.session_state.docserch.as_retriever(search_type="similarity")
    # prompt for retrieving information
    if 'chain_type_kwargs' not in st.session_state:
        st.session_state.chain_type_kwargs = prompt_sector(position, templates)
    # interview history
    if "resume_history" not in st.session_state:
        st.session_state.resume_history = []
        st.session_state.resume_history.append(Message(origin="ai", message="Hello, I am your interivewer today. I will ask you some questions regarding your resume and your experience. Please start by saying hello or introducing yourself. Note: The maximum length of your answer is 4097 tokens!"))
    # token count
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    # memory buffer for resume screen
    if "resume_memory" not in st.session_state:
        st.session_state.resume_memory = ConversationBufferMemory(human_prefix = "Candidate: ", ai_prefix = "Interviewer")
    # guideline for resume screen
    if "resume_guideline" not in st.session_state:
        llm = TextGen(model_url=model_url,**hyperparams)
        st.session_state.resume_guideline = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type_kwargs=st.session_state.chain_type_kwargs, chain_type='stuff',
            retriever=st.session_state.retriever, memory = st.session_state.resume_memory).run("Create an interview guideline and prepare only two questions for each topic. Make sure the questions tests the knowledge")
    # llm chain for resume screen
    if "resume_screen" not in st.session_state:
        llm = TextGen(model_url=model_url,**hyperparams)
        PROMPT = PromptTemplate(
            input_variables=["history", "input"],
            template= """### Instruction:
                Act as an interviewer named GPTInterviewer.

                ### Context:
                You will follow a strict guideline during this conversation. The candidate does not have access to the guideline. 
                Your task is to conduct the interview by asking questions one at a time, waiting for the candidate's answers, and asking necessary follow-up questions. 
                Ensure that you do not repeat or rephrase the same questions. Respond only as an interviewer without providing explanations.

                ### Example:
                Interviewer: What inspired you to pursue this career path?

            {history}
            
            Candidate: {input}
            AI: """)
        st.session_state.resume_screen =  ConversationChain(prompt=PROMPT, llm = llm, memory = st.session_state.resume_memory)
    # llm chain for generating feedback
    if "resume_feedback" not in st.session_state:
        llm = TextGen(model_url=model_url,**hyperparams)
        st.session_state.resume_feedback = ConversationChain(
            prompt=PromptTemplate(input_variables=["history","input"], template=templates.feedback_template),
            llm=llm,
            memory=st.session_state.resume_memory,
        )

def answer_call_back():
    with get_openai_callback() as cb:
        human_answer = st.session_state.answer
        if voice:
            save_wav_file("temp/audio.wav", human_answer)
            try:
                input = transcribe("temp/audio.wav")
                # save human_answer to history
            except:
                st.session_state.resume_history.append(Message("ai", "Sorry, I didn't get that."))
                return "Please try again."
        else:
            input = human_answer
        st.session_state.resume_history.append(
            Message("human", input)
        )
        # OpenAI answer and save to history
        llm_answer = st.session_state.resume_screen.run(input)
        st.session_state.resume_history.append(
            Message("ai", llm_answer)
        )
        st.session_state.token_count += cb.total_tokens
        return llm_answer

if position and resume:
    # initialize session state
    initialize_session_state_resume()
    credit_card_placeholder = st.empty()
    col1, col2 = st.columns(2)
    with col1:
        feedback = st.button("Get Interview Feedback")
    with col2:
        guideline = st.button("Show me interview guideline!")
    chat_placeholder = st.container()
    answer_placeholder = st.container()
    # audio = None
    # if submit email address, get interview feedback immediately
    if guideline:
        st.markdown(st.session_state.resume_guideline)
    if feedback:
        evaluation = st.session_state.resume_feedback.run("please give evaluation regarding the interview")
        st.markdown(evaluation)
        st.download_button(label="Download Interview Feedback", data=evaluation, file_name="interview_feedback.txt")
        st.stop()
    else:
        with answer_placeholder:
            voice: bool = st.checkbox("I would like to speak with AI Interviewer!")
            if voice:
                answer = audio_recorder(pause_threshold=2, sample_rate=44100)
                # st.warning("An UnboundLocalError will occur if the microphone fails to record.")
            else:
                answer = st.chat_input("Your answer")
            if answer:
                st.session_state['answer'] = answer
                answer_call_back()

        with chat_placeholder:
            for answer in st.session_state.resume_history:
                if answer.origin == 'ai':
                    if auto_play:
                        with st.chat_message("assistant"):
                            st.write(answer.message)
                            # st.write(audio)
                    else:
                        with st.chat_message("assistant"):
                            st.write(answer.message)
                else:
                    with st.chat_message("user"):
                        st.write(answer.message)

        credit_card_placeholder.caption(f"""
                        Progress: {int(len(st.session_state.resume_history) / 30 * 100)}% completed.""")
