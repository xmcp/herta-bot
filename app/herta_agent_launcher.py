import os
from dotenv import load_dotenv

load_dotenv()

from herta_agent.graph import graph

if __name__=='__main__':
    #target_title = 'attention is all you need'
    target_title = input('Target Title > ')

    config = {
        "base_url": os.getenv("herta_base_url"),
        "model": os.getenv("herta_model"),
        "api_key": os.getenv("herta_api_key"),
    }

    ret = graph.invoke({
        'target_title': target_title,
    }, config)

    print(ret['report'])