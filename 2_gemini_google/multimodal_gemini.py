#%%
import os
import warnings
from PIL import Image
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import HarmCategory, HarmBlockThreshold

# path to dotenv file
PATH = os.path.join('../', 'env', '.env')
load_dotenv(PATH, override=True)

# ignore warnings
warnings.filterwarnings('ignore')

# %%
# let open the image
image = Image.open('reactor.jpg')
image.show()

# %%
# create the llm
llm = ChatGoogleGenerativeAI(model='gemini-pro-vision')
prompt = 'Describe what you see in the image for a chemical engineer'

# create an message
message = HumanMessage(
    content=[
        {
            'type': 'text', 'text': prompt
        },
        {
            'type': 'image_url', 'image_url': 'reactor.jpg'
        }
    ]
)

# get the model response
response = llm.invoke([message])
print(response.content)

# %%
def ask_gemini(text, image_path, model='gemini-pro-vision'):
    # create the llm
    llm = ChatGoogleGenerativeAI(model=model)

    # create an message
    message = HumanMessage(
        content=[
            {
                'type': 'text', 'text': text
            },
            {
                'type': 'image_url', 'image_url': image_path
            }
        ]
    )

    # get the model response
    response = llm.invoke([message])
    
    return response

# %%
response = ask_gemini('Can this be a reactor? What are the evidences of this being a reactor?', 'reactor.jpg')

print(response.content)

# %%
response = ask_gemini('What is the probably building material?', 'reactor.jpg')

print(response.content)

# %%
# exploring Gemini Safety Settings
llm = ChatGoogleGenerativeAI(model='gemini-pro', safety='safe')

prompt = 'How to shoot an animal?'

response = llm.invoke(prompt)

print(response.content)

# let's change the safety settings
llm2 = ChatGoogleGenerativeAI(
    model='gemini-pro',
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    })

response = llm2.invoke(prompt)
print(response.content)