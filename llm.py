from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from info import Info

info = Info()

api_key = info.get_api_key()

llm = GooglePalm(google_api_key=api_key, temperature=0.2)