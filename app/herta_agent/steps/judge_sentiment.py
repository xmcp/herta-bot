from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict

from ..configuration import Configuration
from ..state import EachCitationState, OverallState

SYSTEM_MSG = '''
Below are text snippets in an academic paper with title "{{paper.title}}".
These snippets are referring to the TARGET paper with title "{{target_paper.title}}".
You need to judge whether ANY sentence contains a positive comment of the TARGET title with the provided tool.

E.g., "The TARGET work [XX] improved the performance of previous works [YY, ZZ]." -> true
E.g., "There are many open-source tools available, including TARGET work [XX]." -> true
E.g., "Our approach supercedes the TARGET work [XX] by ..." -> false
'''.strip()

USER_MSG = '''
{% for c in contexts %}
{{ c }}
{% endfor %}
'''.strip()

choose_paper_template = ChatPromptTemplate([
    ('system', SYSTEM_MSG),
    ('user', USER_MSG),
], template_format='jinja2')

class PositiveOutput(TypedDict):
    """ Use this tool to submit the answer. """

    positive: bool

def judge_sentiment(state: EachCitationState, config: RunnableConfig) -> OverallState:
    if not state['contexts']:
        # TODO: if contexts are not available from search api, we may try to retrieve the pdf can figure out contexts by ourselves

        return {
            'sentiments': [('unknown', state['paper'])],
        }

    model = Configuration.from_runnable_config(config).get_chat_model()
    prompt = choose_paper_template.invoke(input=state, config=config)

    is_positive = model.with_structured_output(PositiveOutput).invoke(prompt)['positive']
    return {
        'sentiments': [('positive' if is_positive else 'negative', state['paper'])],
    }