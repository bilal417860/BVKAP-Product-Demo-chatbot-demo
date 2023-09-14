from langchain import PromptTemplate, OpenAI, LLMChain
import chainlit as cl
from langchain.chat_models import ChatVertexAI
from langchain.schema import AIMessage, HumanMessage
import gradio as gr
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
from langchain.agents import create_sql_agent
from langchain_experimental.sql import SQLDatabaseChain
import os
from langchain import SQLDatabase
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

project = "dastex-genai"
dataset = "pe_firm_scraped"
table = "firm_data"
sqlalchemy_url = f"bigquery://{project}/{dataset}"

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

_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:

{table_info}

The database contains a lot of data, so it is advised to use DISTINCT operators in order to reduce the amount of data. Please use the DISTINCT column operator
where possible.
As the database contains some erroneuous characters, it is better to use LIKE instead of = when filtering.
The function REGEXP does not exist in BigQuery. Instead, use LIKE.

You should include all the information returned, and include it in a bullet point format. 
Do not make up numbers, include only numbers that are returned from the query.
Format any decimal numbers as currency figures.

Question: {input}"""
PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
)

openai_api_key = os.environ["OPENAI_API_KEY"]

template = """Question: {question}

Answer: Let's think step by step."""


@cl.on_chat_start
def main():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key
    )

    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613") # type: ignore
    db = SQLDatabase.from_uri(
        sqlalchemy_url,
        custom_table_info=custom_table_info,
        include_tables=["firm_data"],
    )
    db_chain = SQLDatabaseChain.from_llm(llm, db, prompt=PROMPT, verbose=True)
    tools = [
        Tool(
            name="dbchain",
            func=db_chain.run,
            description="Chat with SQLDB",
        )
    ]

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", agent)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Do any post processing here

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.
    await cl.Message(content=res["output"]).send()
