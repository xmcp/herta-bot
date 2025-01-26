from langgraph.graph import StateGraph, START, END
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
import os

from .configuration import Configuration
from .state import OverallState, InputState
from .steps.query_root_id import query_root_id
from .steps.dispatch_citations import dispatch_citations, continue_to_each_citation
from .steps.judge_sentiment import judge_sentiment
from .steps.make_report import make_report

set_llm_cache(SQLiteCache(database_path='cache/langchain.db'))

builder = StateGraph(OverallState, input=InputState, config_schema=Configuration)

builder.add_node(query_root_id)
builder.add_node(dispatch_citations)
builder.add_node(judge_sentiment)
builder.add_node(make_report)

builder.add_edge(START, 'query_root_id')
builder.add_edge('query_root_id', 'dispatch_citations')
builder.add_conditional_edges('dispatch_citations', continue_to_each_citation, ['judge_sentiment'])
builder.add_edge('judge_sentiment', 'make_report')
builder.add_edge('make_report', END)

graph = builder.compile(debug=os.getenv('herta_debug', '').lower()=='true')
graph.name = 'Herta Agent'
