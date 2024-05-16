#%%
import re
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
#%%
# let's try DuckDuckGo search tool
search = DuckDuckGoSearchRun()

# invoke
output = search.invoke('What is an ideal gas?')
print(output)

#%% 
search = DuckDuckGoSearchResults()
output = search.run('Ideal Gas and Thermodynamics')
print(output)

#%%
# trying the API Wrapper
wrapper = DuckDuckGoSearchAPIWrapper(region='br-br', max_results=10, safesearch='moderate')
search = DuckDuckGoSearchResults(api_wrapper=wrapper, source='scientific-papers')
output = search.run('GenerativeAI AND chemical engineering')

print(output)

#%%
pattern = r'snippet: (.*?), title: (.*?), link: (.*?)\]'
matches = re.findall(pattern, output, re.DOTALL)

for snippet, title, link in matches:
    print(f'Snippet: {snippet}\nTitle: {title}\nLink:{link}')
    print('-' * 50)

#%%
# lets try Wikipedia wrapper
api_wrapper = WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
wiki.invoke({'query': 'thermodynamics'})