from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from info import Info

info = Info()

api_key = info.get_api_key()

llm = GooglePalm(google_api_key=api_key, temperature=0.2)

# database connection
db_user, db_password, db_host, db_name = info.get_db_info()

db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",sample_rows_in_table_info=3)

print(db.table_info)