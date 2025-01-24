from langchain_core.runnables import RunnableConfig

from ..state import OverallState

def make_report(state: OverallState, config: RunnableConfig) -> OverallState:
    sentiments = {}

    for group_name, citation in state['sentiments']:
        sentiments.setdefault(group_name, []).append(citation)

    def ranker(citation):
        return -citation['citationCount']

    for group in sentiments.values():
        group.sort(key=ranker)

    report = []
    for group_name in ['positive', 'negative', 'unknown']:
        citations = sentiments.get(group_name, [])
        report.append(f'\n### {group_name.capitalize()} ({len(citations)})')
        for citation in citations:
            if 'externalIds' in citation and 'DOI' in citation['externalIds']:
                doi = 'doi.org/' + citation['externalIds']['DOI']
            else:
                doi = 'no doi'
            report.append(f'- {citation["title"]} ({citation["year"]}, {citation["citationCount"]} citations, {doi})')

    return {
        'report': '\n'.join(report),
    }