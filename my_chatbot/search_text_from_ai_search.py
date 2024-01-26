import bs4
import requests

from promptflow import tool

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

endpoint = 'https://itemprice-htw.search.windows.net'
admin_key = 'hYabQwITkN4TeYnpgPR7U3yW0UvGkdEYM2HSwf5nFGAzSeDYSPgs'
index = 'azureblob-index'
search_client = SearchClient(endpoint=endpoint,
                      index_name=index,
                      api_version="2021-04-30-Preview",
                      credential=AzureKeyCredential(admin_key))

@tool
def search_text_from_ai_search(text: str):
    results = search_client.search(search_text='''{
    "search": "떡"
}''', top=3, include_total_count=True)
    output = [r for r in results]
    # intermediate_output = " ".join(output)
    # print(intermediate_output)
    return output
