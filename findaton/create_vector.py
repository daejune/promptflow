import os
import re
import requests
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
import tiktoken
from openai import AzureOpenAI

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    HnswParameters,
    ExhaustiveKnnAlgorithmConfiguration,
    VectorSearchProfile,
    VectorSearchAlgorithmMetric,
    ExhaustiveKnnParameters,
    SearchableField,
    SearchFieldDataType,
    SearchField,
    SimpleField,
)

endpoint = "https://findaton-legal-example.search.windows.net"
admin_key = "aliTJmqKCW2Yb4eB6m8tUePcXIik0AbqkNl6Z5yg2XAzSeA2a9gl"
index = "azureblob-index"
search_client = SearchClient(
    endpoint=endpoint,
    index_name=index,
    api_version="2021-04-30-Preview",
    credential=AzureKeyCredential(admin_key),
)

pd.options.mode.chained_assignment = None  # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#evaluation-order-matters


# Configure the vector search configuration
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="myHnsw",
            kind=VectorSearchAlgorithmKind.HNSW,
            parameters=HnswParameters(
                m=4,
                ef_construction=400,
                ef_search=500,
                metric=VectorSearchAlgorithmMetric.COSINE,
            ),
        ),
        ExhaustiveKnnAlgorithmConfiguration(
            name="myExhaustiveKnn",
            kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
            parameters=ExhaustiveKnnParameters(
                metric=VectorSearchAlgorithmMetric.COSINE
            ),
        ),
    ],
    profiles=[
        VectorSearchProfile(
            name="myHnswProfile",
            algorithm_configuration_name="myHnsw",
        ),
        VectorSearchProfile(
            name="myExhaustiveKnnProfile",
            algorithm_configuration_name="myExhaustiveKnn",
        ),
    ],
)


# s is input text
def normalize_text(s, sep_token=" \n "):
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r". ,", "", s)
    # remove all instances of multiple spaces
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.strip()

    return s


def generate_embeddings(
    text, model="text-embedding-ada-002"
):  # model = "deployment_name"
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def generate_ads(
    keywords, model="gpt-35-turbo", temperature=0.7, max_response_tokens=256
):
    return (
        client.chat.completions.create(
            model=model,  # model = "deployment_name".
            temperature=temperature,
            max_tokens=max_response_tokens,
            messages=[
                {
                    "role": "system",
                    "content": "너는 한국에서 활동하는 앱서비스 광고전문가 입니다. 키워드에 맞는 광고 문구를 만들어야 합니다. 광고 문구를 사용하는 회사는 핀다라는 앱서비스 회사입니다. 핀다의 서비스는 대출비교, 대환대출, 나의 신용점수관리, 나의 자산관리, 포인트 받기 기능을 가지고 있습니다.",
                },
                {"role": "user", "content": keywords},
                {"role": "assistant", "content": "핀다에서 대출비교를 하면 10% 할인 혜택을 드립니다."},
            ],
        )
        .choices[0]
        .message.content
    )


def search_vector(fields, data):
    print(data)
    vector_query = VectorizedQuery(
        vector=generate_embeddings(data), k_nearest_neighbors=3, fields=fields
    )

    return search_client.search(search_text=None, vector_queries=[vector_query], top=3)


if __name__ == "__main__":
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-07-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    file_name = "test" + ".csv"
    df = pd.read_csv(os.path.join(os.getcwd(), file_name))

    # print(generate_embeddings ('배가 고픕니다.', model = 'text-embedding-ada-002'))
    # df['content_vector'] = df["content"].apply(lambda x : generate_embeddings (client=client, text=x, model = 'text-embedding-ada-002')) # model should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    # df.to_csv(os.path.join(os.getcwd(),'result.csv'))

    max_response_tokens = 512
    token_limit = 4096
    result = [generate_ads("벚꽃, 대출") for _ in range(0, 10)]
    sim = [search_vector("contents_vector", result[_]) for _ in range(0, len(result))]
    print(sim)
