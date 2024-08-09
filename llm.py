from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from info import Info
from few_shots import few_shots

info = Info()

api_key = info.get_api_key()

llm = GooglePalm(google_api_key=api_key, temperature=0.2)

# database connection
db_user, db_password, db_host, db_name = info.get_db_info()

db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",sample_rows_in_table_info=3)

print(db.table_info)

# database chain
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# test query
qns1 = db_chain("How many t-shirts do we have left for nike in extra small size and white color?")

# qns2 = db_chain.run("How much is the price of the inventory for all small size t-shirts?")
qns2 = db_chain.run("SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'")

# qns3 = db_chain.run("If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue our store will generate (post discounts)?")

sql_code = """
select sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
(select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi'
group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
 """

qns3 = db_chain.run(sql_code)

qns4 = db_chain.run("SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Levi'")

qns5 = db_chain.run("How many white color Levi's t shirts we have available?")

qns5 = db_chain.run("SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'")

### Few Shot Learning

# generating embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

to_vectorize = [" ".join(example.values()) for example in few_shots]

vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)

example_selector.select_examples({"Question": "How many Adidas T shirts I have left in my store?"})

# my sql based instruction prompt
mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use CURDATE() function to get the current date, if the question involves "today".

Use the following format:

Question: Question here
SQLQuery: Query to run with no pre-amble
SQLResult: Result of the SQLQuery
Answer: Final answer here

No pre-amble.
"""

from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt

print(PROMPT_SUFFIX)

from langchain.prompts.prompt import PromptTemplate

example_prompt = PromptTemplate(
    input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
    template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
)

print(_mysql_prompt)