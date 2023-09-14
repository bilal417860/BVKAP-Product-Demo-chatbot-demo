from flask import Flask, session, request, send_file, jsonify, render_template, Response, stream_with_context
import requests
#from flask_cors import CORS
import os
import json
import time

app = Flask(__name__)
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

project = "dastex-genai"
dataset = "pe_firm_scraped"
table = "firm_data"
sqlalchemy_url = f'bigquery://{project}/{dataset}'

from langchain import SQLDatabase
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

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

The function REGEXP does not exist in BigQuery. Instead, use LIKE.

You should include all the information returned, and include it in a bullet point format. 
Do not make up numbers, include only numbers that are returned from the query.

Question: {input}"""
PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
)

openai_api_key = os.environ['OPENAI_API_KEY']

llm = ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0, openai_api_key=openai_api_key)

#llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613") # type: ignore
db = SQLDatabase.from_uri(sqlalchemy_url, custom_table_info=custom_table_info, include_tables=['firm_data'])
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
    memory=memory
)

def predict(message, history=None):
    history_langchain_format = []
    """
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    """
    #gpt_response = llm(history_langchain_format)
    gpt_response = agent({'input': message})
    return gpt_response['output']

my_path_to_server01 = 'http://localhost:5173/'

@app.route("/generate", methods=['POST'])
def generate():
    """Example Hello World route."""
    data = request.get_json()
    r = requests.get(my_path_to_server01, stream=True)
    query = data["inputs"]
    result = predict(query)
    #answer = {"generated_text": '<|assistant|>' + result}
    answer = '<|assistant|>' + result
    _resp = requests.get(my_path_to_server01, stream=True)
    _resp.raise_for_status()

    resp = Response(
        response=stream_with_context(_resp.iter_content(chunk_size=1024*10)),
        status=200,
        content_type=_resp.headers["Content-Type"],
        direct_passthrough=False)
    rep = Response(json.dumps(answer), mimetype='application/json')
    @stream_with_context
    def generate():
        print('1')
        yield "<|assistant|>Hello"
        time.sleep(1) # Simulate some delay
        print('2')
        yield "<|assistant|>How"
        time.sleep(1)
        print('3')
        yield "<|assistant|>Can"
        time.sleep(1)
        print('4')
        yield "Streaming data\n"
        time.sleep(1)
        print('5')

    return Response(generate(), content_type='text/event-stream')

    return jsonify(answer)

@app.route("/summarize", methods=['POST'])
def summarize():
	"""Example Hello World route."""
	data = request.get_json()
	r = requests.get(my_path_to_server01, stream=True)
	query = data["inputs"]
	result = predict(query)
	#answer = {"generated_text": '<|assistant|>' + result}
	answer = '<|assistant|>' + result
	_resp = requests.get(my_path_to_server01, stream=True)
	_resp.raise_for_status()

	resp = Response(
		response=stream_with_context(_resp.iter_content(chunk_size=1024*10)),
		status=200,
		content_type=_resp.headers["Content-Type"],
		direct_passthrough=False)
	rep = Response(json.dumps(answer), mimetype='application/json')
	@stream_with_context
	def generate():
		print('1')
		yield "<|prompter|>Hello"
		time.sleep(1) # Simulate some delay
		print('2')
		yield "<|assistant|>How"
		time.sleep(1)
		print('3')
		yield "<|prompter|>Can"
		time.sleep(1)
		print('4')
		yield "Streaming data\n"
		time.sleep(1)
		print('5')

	answer = {'user': 'Test'}
    
	return jsonify(answer)

    #return Response(generate(), content_type='text/event-stream')

if __name__ == '__main__':
	app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))