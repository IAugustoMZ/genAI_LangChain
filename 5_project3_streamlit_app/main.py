import os
import streamlit as st
from services.chat import ChatAcademicAdvisor
from repository.vector_storage import ChromaVectorStorage

# create the chatbot
chat = ChatAcademicAdvisor()

st.title('ChatAdvisor: A Chatbot for Acedmic Advising')

st.subheader('LLM Question and Answering app about your paper')

with st.sidebar:

    # get the user input
    uploaded_file = st.file_uploader("Upload your paper", type=['pdf'])

    file_name = st.text_input('File name:', value='')
    chunk_size = st.number_input('Chunk size', min_value=100, max_value=2048, value=256)
    chunk_overlap = st.number_input('Chunk overlap', min_value=0, max_value=512, value=0)
    top_k = st.number_input('Top K retrieved: ', min_value=0, max_value=512, value=5)
    

    add_data = st.button('Add File')

    # crate file path
    file_path = f'repository/data/{file_name}.pdf'

    if uploaded_file and add_data:
        with st.spinner('Loading and preparing your document'):
            
            # get the binary data
            bytes_data = uploaded_file.read()

            # write the file in disk
            with open(file_path, 'wb') as f:
                f.write(bytes_data)

            # load the document as string
            data = ChromaVectorStorage.load_document(file_path)

            # remove the documento to save space
            os.remove(file_path)

            # chunk the data
            chunks = ChromaVectorStorage.chunk_data(
                data,
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap)
            )

            # insert the embeddings
            vector_store = ChromaVectorStorage().create_embedding_store(chunks)

            # save it in the session id
            st.session_state.vs = vector_store

            st.success('File uploaded and processed successfully!')

# let the user ask questions
q = st.text_input('Ask a question about your paper: ')

if q and ('vs' in st.session_state):

    # get the answer
    answer = chat.ask_and_get_questions(q, st.session_state.vs, top_k=int(top_k))

    st.text_area('Adivsor Answer:', value=answer)

        


