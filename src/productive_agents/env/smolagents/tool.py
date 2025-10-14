from dataclasses import dataclass
from typing import Any, Dict, Optional

from smolagents.tools import Tool

class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {"answer": {"type": "any", "description": "The final answer to the problem"}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        return answer

class WikipediaRetrieverTool(Tool):
    name = "wikipedia_search"
    description = "Uses semantic search to retrieve the parts of 2018 wikipedia that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        },
        "n_results": {
            "type": "integer",
            "description": "The number of results to return. Minumum is 3. Maximum is 10."
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__()
        if "port" in kwargs.keys():
            self.port = kwargs["port"]
        else:
            self.port = "8005"
        self.url = f"http://127.0.0.1:{self.port}/retrieve"

    def forward(self, query: str, n_results: int) -> str:
        # return "Test Tool"
        import requests

        if n_results < 3:
            n_results = 3
        if n_results > 10:
            # Limit to 10 results
            n_results = min(n_results, 10)  # Ensure max_results does not exceed 10

        assert isinstance(query, str), "Your search query must be a string"
        payload = {
            "queries": [query],
            "topk": n_results,
            "return_scores": True
        }

        # Send POST request
        response = requests.post(self.url, json=payload)

        # Raise an exception if the request failed
        response.raise_for_status()

        # Get the JSON response
        retrieved_data = response.json()
        docs = retrieved_data["result"][0]

        return "Retrieved documents:" + "".join(
            [
                f"\n\n[Document {str(i)}]\n" + doc["document"]["contents"]
                for i, doc in enumerate(docs)
            ]
        )