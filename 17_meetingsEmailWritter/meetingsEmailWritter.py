import os
import warnings
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# define path to the .env file
env_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    '.env'
)
load_dotenv(dotenv_path=env_path, override=True)

# ignore the warnings
warnings.filterwarnings("ignore")

# create the chatbot
chatbot = ChatGroq(
    model='llama-3.1-70b-versatile',
    temperature=0.7,
    max_tokens=None,
    max_retries=2,
)

template = """
Below is a draft of text that might be poorly written. You, as a redaction expert should rewrite it in a more professional way. 

Specifically, your goal is:
- to transform the bullet point texts in a professional email, observing a specific tone (which depends on the receiver role) and the email length.

Please start with the appropriate salutation and end with a closing.

DRAFT: {draft}
RECEIVER ROLE: {receiver_role}
EMAIL LENGTH: {email_length}

Rewritten email:
"""

# create the prompt
prompt = ChatPromptTemplate.from_template(
    template=template,
    placeholders=['draft', 'receiver_role', 'email_length']
)

# page title and header
st.set_page_config(page_title='Meetings Email Writer', page_icon=':email:')
st.header('Put your meetings bullet points and get a professional email!')

# get the bullet points
st.subheader('Bullet Points')
bullet_points = st.text_area('Write the bullet points here')

# get the receiver role
col1, col2 = st.columns(2)
with col1:
    st.subheader('Receiver Role')
    receiver_role = st.selectbox('Select the receiver role', ['Manager', 'Client', 'Coworker', 'CEO', 'Friend'])

with col2:
    st.subheader('Email Length')
    email_length = st.selectbox('Select the email length', ['Short', 'Medium', 'Long'])

if bullet_points:

    prompt_with_draft = prompt.format(
        draft=bullet_points,
        receiver_role=receiver_role,
        email_length=email_length
    )

    # email
    email = chatbot.invoke(bullet_points)

    # show the email
    st.subheader('Professional Email')
    st.write(email.content)