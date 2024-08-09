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