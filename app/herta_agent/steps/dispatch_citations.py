from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

from ..state import OverallState, EachCitationState
from ..api import search_citation

MAX_CITATION_SEARCH = 100
MAX_CITATION_RETURN = 20

def dispatch_citations(state: OverallState, config: RunnableConfig) -> OverallState:
    citations = search_citation(state['target_paper']['paperId'], MAX_CITATION_SEARCH)

    def ranker(citation):
        # prefer citations with valid contexts, then latest year, then most cited
        return int(len(citation[1]) == 0), -citation[0]['year'], -citation[0]['citationCount']

    citations.sort(key=ranker)
    return {
        'citations': citations[:MAX_CITATION_RETURN],
    }

def continue_to_each_citation(state: OverallState) -> list[Send]:
    return [
        Send('judge_sentiment', EachCitationState(
            target_paper=state['target_paper'],
            paper=c[0],
            contexts=c[1],
        )) for c in state['citations']
    ]