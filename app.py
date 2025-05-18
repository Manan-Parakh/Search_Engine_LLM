import streamlit as st
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_groq import ChatGroq

# Tools Setup
wiki_wrapper = WikipediaAPIWrapper(doc_content_chars_max=250, top_k_results=1)
wikipedia = WikipediaQueryRun(api_wrapper=wiki_wrapper,
                              name= 'wikipedia',
                              description='Use this tool to find out about names, places, definitions and historical facts.')

arxiv_wrapper = ArxivAPIWrapper(doc_content_chars_max=250, top_k_results=1)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper,
                      name = 'arxiv',
                      description='Use this tool to find research papers.')

search = DuckDuckGoSearchRun(name='search',
                             description="Use this tool to search the web")

tools = [wikipedia,arxiv,search]

# Streamlit Page
st.set_page_config(page_title="Search Engine", page_icon="🔍")
st.title("Langchain - Chat with Search Tools")

# Get the groq api key
st.sidebar.title('Enter the Groq API Key')
groq_api_key = st.sidebar.text_input('Enter the Groq API Key:', type='password')

if not groq_api_key:
    st.warning('Please enter GROQ API Key to proceed!')
    st.stop()

# Cache the model
@st.cache_resource
def get_agent(groq_api_key):
    llm = ChatGroq(api_key=groq_api_key, model_name="gemma2-9b-it", streaming=True)
    return initialize_agent(agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                            llm=llm,
                            tools=tools,
                            handle_parsing_error = True,
                            verbose = True)

agent = get_agent(groq_api_key)

# Session Management
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant", "content": "Hey! How can I help you today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# Prompt Handler
if prompt := st.chat_input('Ask me anything...'):
    st.session_state.messages.append({'role':"user", "content":prompt})
    st.chat_message('user').write(prompt)

    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            # Create the callback instance
            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)

            try:
                response = agent.run(prompt, callbacks=[st_cb])
            except Exception as e:
                response = f"An Error Occured: {e}"
            
            st.write(response)
            st.session_state.messages.append({'role':"assistant","content":response})