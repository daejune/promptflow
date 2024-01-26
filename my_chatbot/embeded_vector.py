import openai
import bs4
from promptflow import tool


openai.api_type = "azure"
openai.api_version = "2023-06-01-preview"
openai.api_key = "2014112a795a4f0ebcd43201a9875574".strip()
openai.api_base = "https://findaton-team-10.openai.azure.com/".strip()


model = 'text-embedding-ada-002'

@tool
def embeded_vector(input: str):
    soup = bs4.BeautifulSoup(openai.Embedding.create(
        input=input, engine=model
    )["data"][0]["embedding"], "html.parser")
    soup.prettify()
    print(soup.prettify())
    return soup.get_text()[:2000]
