import os
import re
import textwrap
from typing import Optional

import networkx as nx
import matplotlib.pyplot as plt
from arango import ArangoClient
from flask import Flask, render_template, request
from groq import Groq
from langchain.chains import ArangoGraphQAChain
from langchain_community.graphs import ArangoGraph
from langchain_groq import ChatGroq

# =====================================
#  Flask Application Initialization
# =====================================
app = Flask(__name__)

# =====================================
#  Environment Variables
# =====================================
os.environ["GROQ_API_KEY"] = "your_api_key_here"

# =====================================
#  Database Connection (ArangoDB)
# =====================================
try:
    client = ArangoClient(hosts="http://localhost:8529")
    db = client.db("_system", username="root", password="")
    arango_graph = ArangoGraph(db)
except Exception as e:
    raise ConnectionError(f"Failed to connect to ArangoDB: {e}")

# Initialize Groq Client for LLM-based processing
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in environment variables.")

llm_client = Groq(api_key=groq_api_key)

# =====================================
#  Tool 1: Convert Query → AQL → Execute → Return Result
# =====================================
def text_to_aql_to_text(query: str) -> str:
    """
    Converts a natural language query into an AQL query, executes it,
    and returns the result in human-readable text.

    Args:
        query (str): User input query.

    Returns:
        str: Processed query result in text format.
    """
    try:
        llm = ChatGroq(temperature=0, model_name="qwen-2.5-32b")
        chain = ArangoGraphQAChain.from_llm(
            llm=llm,
            graph=arango_graph,
            verbose=True,
            allow_dangerous_requests=True,
        )
        result = chain.invoke(query)
        return str(result.get("result", "No result found"))
    except Exception as e:
        return f"Error processing query: {e}"

# =====================================
#  Tool 2: Convert Query → AQL → Graph → Visualize
# =====================================
def text_to_aql_to_nxalgorithm(query: str) -> Optional[str]:
    """
    Processes a natural language query, extracts relevant data,
    builds a NetworkX graph, and generates a visualization.

    Args:
        query (str): User input query.

    Returns:
        Optional[str]: Generated Python code for visualization.
    """
    try:
        # Generate AQL and fetch data
        llm = ChatGroq(temperature=0, model_name="qwen-2.5-32b")
        chain = ArangoGraphQAChain.from_llm(
            llm=llm,
            graph=arango_graph,
            verbose=True,
            allow_dangerous_requests=True,
        )
        result = chain.invoke(query)
        result_text = str(result.get("result", ""))

        # Use LLM to generate NetworkX visualization code
        chat_completion = llm_client.chat.completions.create(
            model="qwen-2.5-coder-32b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a Data Visualization Assistant. Generate Python code "
                        f"for visualizing NetworkX graphs. Query: {query}, Data: {result_text}. "
                        f"Ensure the output saves the plot as 'static/plot.png'."
                    ),
                }
            ],
        )

        # Extract and clean generated Python code
        generated_code = chat_completion.choices[0].message.content
        match = re.search(r"python\s*(.*?)\s*", generated_code, re.DOTALL)
        return match.group(1).strip() if match else None

    except Exception as e:
        return f"Error generating visualization: {e}"

# =====================================
#  Query Handling Function
# =====================================
def query_graph(query: str) -> str:
    """
    Determines the appropriate tool based on the query type:
      - If the query requests visualization, use `text_to_aql_to_nxalgorithm`
      - Otherwise, use `text_to_aql_to_text`

    Args:
        query (str): User input query.

    Returns:
        str: Processed query result or visualization script.
    """
    return text_to_aql_to_nxalgorithm(query) if any(
        keyword in query.lower() for keyword in ["visualize", "show"]
    ) else text_to_aql_to_text(query)

# =====================================
#  Flask Routes
# =====================================
@app.route("/")
def home() -> str:
    """Renders the homepage with an input form."""
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def handle_query() -> str:
    """
    Handles text-based queries, processes them,
    and displays the result.

    Returns:
        str: Rendered HTML with the result.
    """
    user_query = request.form["query"]
    result = text_to_aql_to_text(user_query)
    return render_template("index.html", result=result)

@app.route("/visualize", methods=["POST"])
def visualize_query() -> str:
    """
    Handles visualization queries, generates a graph,
    and displays the plot.

    Returns:
        str: Rendered HTML with the visualization.
    """
    user_query = request.form["query"]
    result_code = text_to_aql_to_nxalgorithm(user_query)

    if result_code:
        try:
            exec(textwrap.dedent(result_code))  # Execute the generated visualization code
        except Exception as e:
            return render_template("index.html", error=f"Error executing visualization code: {e}")

    return render_template("index.html", plot="static/plot.png")

# =====================================
#  Run Flask Application
# =====================================
if __name__ == "__main__":
    app.run(debug=False)
