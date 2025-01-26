from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, TypedDict

from ..state import OverallState
from ..api import search_paper
from ..configuration import Configuration

SYSTEM_MSG = '''
Below are multiple papers.
Choose the index of the paper that EXACTLY match the title "{{title}}".
If none of them match, submit null.
'''.strip()

USER_MSG = '''
{% for paper in papers %}
{{loop.index}}. {{paper.title}} ({{paper.year}}, {{paper.citationCount}} citations)
{% endfor %}
'''.strip()

choose_paper_template = ChatPromptTemplate([
    ('system', SYSTEM_MSG),
    ('user', USER_MSG),
], template_format='jinja2')

class IndexOutput(TypedDict):
    """ Use this tool to submit the answer. """

    index: Optional[int]

def query_root_id(state: OverallState, config: RunnableConfig) -> OverallState:
    results = search_paper(state['target_title'])
    if len(results) == 0:
        raise ValueError('no paper found')
    elif len(results)>1:
        # ask the llm to choose one

        model = Configuration.from_runnable_config(config).get_chat_model()
        prompt = choose_paper_template.invoke(input=dict(
            title=state['target_title'],
            papers=results,
        ), config=config)

        chosen_idx: int = model.with_structured_output(IndexOutput).invoke(prompt)['index']
        if chosen_idx is None:
            raise ValueError('no matching paper')
        if not 0 <= chosen_idx < len(results):
            raise ValueError(f'invalid index {chosen_idx} (count: {len(results)})')
        results = [results[chosen_idx]]

    return {
        'target_paper': results[0],
    }