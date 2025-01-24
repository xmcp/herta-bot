import requests_cache

from .state import PaperMetadata

session = requests_cache.CachedSession('cache/http.db', expire_after=86400)

# https://api.semanticscholar.org/api-docs/
NECESSARY_FIELDS = ['title', 'paperId', 'authors', 'venue', 'year', 'externalIds', 'citationCount']
SEARCHED_FIELDS = ['Computer Science']

def search_paper(title: str) -> list[PaperMetadata]:
    response = session.get('https://api.semanticscholar.org/graph/v1/paper/search/match', params={
        'query': title,
        'fields': ','.join(NECESSARY_FIELDS),
        'fieldsOfStudy': SEARCHED_FIELDS,
    })
    response.raise_for_status()
    return response.json()['data']

def search_citation(paper_id: str, limit: int) -> list[tuple[PaperMetadata, list[str]]]:
    response = session.get(f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations', params={
        'fields': ','.join(['contexts'] + ['citingPaper.'+f for f in NECESSARY_FIELDS]),
        'limit': limit,
    })
    response.raise_for_status()
    return [(d['citingPaper'], d['contexts']) for d in response.json()['data']]