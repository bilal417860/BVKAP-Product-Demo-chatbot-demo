from langchain.chat_models import ChatVertexAI
from langchain.schema import AIMessage, HumanMessage
import gradio as gr
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
from langchain.agents import create_sql_agent
from langchain_experimental.sql import SQLDatabaseChain

project = "dastex-genai"
dataset = "pe_firm_scraped"
table = "firm_data"
sqlalchemy_url = f'bigquery://{project}/{dataset}'

custom_table_info = {
    "firm_data": """
CREATE TABLE `firm_data` (
	`FirmName` STRING OPTIONS(description='Firm Name'), 
	`EquityType` STRING OPTIONS(description='Equity Type'), 
	`GeographicalFocus` STRING OPTIONS(description='Geographical Focus'), 
	`Industrysector` STRING OPTIONS(description='Industry sector'), 
	`InvestmentReasons` STRING OPTIONS(description='Investment Reasons'), 
	`InvestmentType` STRING OPTIONS(description='Investment Type'), 
	`InvestorTypes` STRING OPTIONS(description='Investor Types'), 
	`ManagingDirectorsPartners` STRING OPTIONS(description='Managing Directors / Partners'), 
	`Offices` STRING OPTIONS(description='Offices'), 
	`Shareholders` STRING OPTIONS(description='Shareholders'), 
	`ActivesinceinGermany` STRING OPTIONS(description='Active since (in Germany)'), 
	`Capital` FLOAT64 OPTIONS(description='Capital'), 
	`Capitalyear` STRING OPTIONS(description='Capital year'), 
	`FundCount` STRING OPTIONS(description='Fund Count'), 
	`Photo` STRING OPTIONS(description='Photo'), 
	`PortfolioCompanies` STRING OPTIONS(description='Portfolio Companies'), 
	`Professionals` STRING OPTIONS(description='Professionals'), 
	`Equityholdings` STRING OPTIONS(description='Equity holdings'), 
	`Equityholdingsyear` STRING OPTIONS(description='Equity holdings year'), 
	`FundYear` STRING OPTIONS(description='Fund Year'), 
	`Investmentcases` STRING OPTIONS(description='Investment cases'), 
	`Investors` STRING OPTIONS(description='Investors'), 
	`PortfoliocompanyYear` STRING OPTIONS(description='Portfolio company Year'), 
	`Staffcount` STRING OPTIONS(description='Staff count'), 
	`MinimumEquityInvestmentRange` FLOAT64 OPTIONS(description='Minimum Equity / Investment Range'), 
	`MaximumEquityInvestmentRange` FLOAT64 OPTIONS(description='Maximum Equity / Investment Range'), 
	`Minimumturnoverofcompanies` FLOAT64 OPTIONS(description='Minimum turnover of companies'), 
	`Maximumturnoverofcompanies` FLOAT64 OPTIONS(description='Maximum turnover of companies'), 
	`Minimumtransactionvolume` FLOAT64 OPTIONS(description='Minimum transaction volume'), 
	`Maximumtransactionvolume` FLOAT64 OPTIONS(description='Maximum transaction volume')
)
    """
}

from langchain.prompts.prompt import PromptTemplate

_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:

{table_info}

The function REGEXP does not exist in BigQuery. Instead, use LIKE.

You should include all the information returned, and include it in a bullet point format. 
Do not make up numbers, include only numbers that are returned from the query.
When there are numbers in the output, you should display them in currency format.

Question: {input}"""
PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
)

llm = VertexAI(
            model_name="text-bison@001",
            max_output_tokens=1024,
            temperature=0.1,
            #top_p=0.8,
            #top_k=40,
        )

db = SQLDatabase.from_uri(sqlalchemy_url, custom_table_info=custom_table_info, include_tables=['firm_data'])
db_chain = SQLDatabaseChain.from_llm(llm, db, prompt=PROMPT, verbose=True, use_query_checker=True,  return_intermediate_steps=False, top_k=20)

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    #gpt_response = llm(history_langchain_format)
    gpt_response = db_chain.run(message)
    return gpt_response

gr.ChatInterface(predict).launch()