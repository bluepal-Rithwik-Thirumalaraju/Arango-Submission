
#Standard Library (Built-in Python modules)
import os
import re
import textwrap
from typing import Optional

#Data Processing & Visualization
import networkx as nx
import matplotlib.pyplot as plt

#Database & Graph Management
from arango import ArangoClient
from langchain.chains import ArangoGraphQAChain
from langchain_community.graphs import ArangoGraph

# AI & NLP (Natural Language Processing)
from groq import Groq
from langchain_groq import ChatGroq

#Web Framework
from flask import Flask, render_template, request


# =====================================
#  Flask Application Initialization
# =====================================
"""
This section initializes the Flask application, which serves as the web interface
for interacting with the ArangoDB database and performing queries.
"""
app = Flask(__name__)


# =====================================
#  Environment Variables
# =====================================
"""
This section sets up environment variables, such as the GROQ API key,
which is required for interacting with the Groq API for natural language processing.
"""
os.environ["GROQ_API_KEY"] = "your_api_key_here"


# =====================================
#  Database Connection (ArangoDB)
# =====================================
"""
This section establishes a connection to the ArangoDB database.
It initializes the ArangoGraph object, which is used to interact with the database.
"""
try:
    client = ArangoClient(hosts="http://localhost:8529")
    db = client.db("_system", username="root", password="")
    arango_graph = ArangoGraph(db)
except Exception as e:
    raise ConnectionError(f"Failed to connect to ArangoDB: {e}")


# =====================================
#  Initialize Groq Client for LLM-based Processing
# =====================================
"""
This section initializes the Groq client, which is used for natural language processing
and generating AQL queries from user input.
"""
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in environment variables.")

llm_client = Groq(api_key=groq_api_key)


# =====================================
#  Tool 1: Convert Query → AQL → Execute → Return Result
# =====================================
"""
This function converts a natural language query into an AQL query, executes it,
and returns the result in human-readable text.
"""
def text_to_aql_to_text(query: str) -> str:
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
"""
This function processes a natural language query, extracts relevant data,
builds a NetworkX graph, and generates a visualization.
"""
def text_to_aql_to_nxalgorithm(query: str) -> Optional[str]:
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
        chat_completion = client.chat.completions.create(
                model="qwen-2.5-coder-32b",
                messages=[
                    {  
                        "role": "system",
                        "content": (
    
                            "You are a Data Visualization Assistant. Your task is to generate only Python code for networkx visualizations. The below Rules to Follow "
                            f"based on user query :{query} from our data {result}. The output must be pure Python code, without any text, imports, explanations."

                            "### Rules to Follow"
                            f"1. Generate a working visualization code for {result} data using the exact format and structure from the provided examples."
                            "2. After generating the code, verify it thoroughly to ensure there are no errors and that it functions as expected."
                            "3. Ensure the final visualization uses plt.savefig('static/plot.png') Do not use plt.show() or return statements."

                            "### Examples "

                            "*Example 1:* "
                            "*query:* 'list all the products of users' "
                            "*result:* 'Summary:'"
                                        'The list of products purchased by users includes the following items:'
                                        '- Wireless Mouse (product ID: products/prod1) was bought twice by one user and once by another, each at a price of $29.99.'
                                        '- Another product (product ID: products/prod2) was purchased once by one user and three times by another, each at a price of $12.50.'

                                        'This summary provides a breakdown of the products that have been purchased by users, detailing the product ID, quantity, and price for each purchase.'

                            "*Expected Model Output:* "

                            "python "
                            """
                            import networkx as nx
                            import matplotlib.pyplot as plt

                            # Create a directed graph
                            G_clean = nx.DiGraph()

                            # Product details
                            product_details = {
                                "products/prod1": ("Wireless Mouse", 29.99),
                                "products/prod2": ("Ceramic Mug", 12.50)
                            }

                            # Updated product nodes with names and prices
                            product_nodes = {pid: f'{name}\n(${price})' for pid, (name, price) in product_details.items()}

                            # Define users and purchases
                            users = ["User1", "User2"]
                            purchases = [("User1", "products/prod1", 2),
                                        ("User2", "products/prod1", 1),
                                        ("User1", "products/prod2", 1),
                                        ("User2", "products/prod2", 3)]

                            # Add product nodes
                            for pid, label in product_nodes.items():
                                G_clean.add_node(label, color="lightgreen")

                            # Add user nodes and purchase edges
                            for user in users:
                                G_clean.add_node(user, color="lightblue")  # Users as blue nodes

                            for user, pid, qty in purchases:
                                G_clean.add_edge(user, product_nodes[pid], weight=qty)

                            # Update node colors
                            node_colors = [G_clean.nodes[n]["color"] for n in G_clean.nodes]

                            # Draw the graph
                            plt.figure(figsize=(7, 5))
                            pos = nx.spring_layout(G_clean, seed=42)  # Fixed seed for consistent layout

                            # Draw nodes and edges
                            nx.draw(G_clean, pos, with_labels=True, node_color=node_colors, edge_color="gray", node_size=2500, font_size=9)
                            edge_labels = {(u, v): G_clean[u][v]['weight'] for u, v in G_clean.edges}
                            nx.draw_networkx_edge_labels(G_clean, pos, edge_labels=edge_labels, font_size=9)

                            plt.title("Network Graph: User Product Purchases")
                            plt.savefig("static/plot.png")
                            """

                            "**Example 2:** "
                            "**query:** 'list all the patients with diseases' "
                            "**result:** 'Summary: The list of patients who have been diagnosed with diseases includes the following individuals: 1. Drona 1, a 51-year-old male with blood type A-. 2. Ashwatthama 2, a 47-year-old male with blood type A-. 3. Parshurama 3, an 8-year-old male with blood type AB-. 4. Daksha 4, an 89-year-old female with blood type A-. 5. Hanuman 5, a 46-year-old female with blood type O+. 6. Indra 6, an 81-year-old male with blood type O+. Please note that some patients appear more than once in the list because they are associated with multiple diseases.'"
                            "**Expected Model Output:** "

                            "python "
                            """
                            import networkx as nx
                            import matplotlib.pyplot as plt
                            
                            # Create a directed graph
                            G_clean = nx.DiGraph()
                            
                            # Patient details
                            patient_details = {
                                "Drona 1": {"age": 51, "gender": "male", "blood_type": "A-"},
                                "Ashwatthama 2": {"age": 47, "gender": "male", "blood_type": "A-"},
                                "Parshurama 3": {"age": 8, "gender": "male", "blood_type": "AB-"},
                                "Daksha 4": {"age": 89, "gender": "female", "blood_type": "A-"},
                                "Hanuman 5": {"age": 46, "gender": "female", "blood_type": "O+"},
                                "Indra 6": {"age": 81, "gender": "male", "blood_type": "O+"}
                            }
                            
                            # Define diseases and patient associations
                            diseases = ["Disease1", "Disease2"]
                            associations = [("Drona 1", "Disease1"),
                                            ("Ashwatthama 2", "Disease1"),
                                            ("Parshurama 3", "Disease2"),
                                            ("Daksha 4", "Disease1"),
                                            ("Hanuman 5", "Disease2"),
                                            ("Indra 6", "Disease1")]
                            
                            # Add patient nodes with details
                            for patient, details in patient_details.items():
                                G_clean.add_node(patient, age=details["age"], gender=details["gender"], blood_type=details["blood_type"], color="lightblue")
                            
                            # Add disease nodes
                            for disease in diseases:
                                G_clean.add_node(disease, color="lightgreen")
                            
                            # Add patient-disease associations
                            for patient, disease in associations:
                                G_clean.add_edge(patient, disease)
                            
                            # Update node colors
                            node_colors = [G_clean.nodes[n]["color"] for n in G_clean.nodes]
                            
                            # Draw the graph
                            plt.figure(figsize=(9, 6))
                            pos = nx.spring_layout(G_clean, seed=42)  # Fixed seed for consistent layout
                            
                            # Draw nodes and edges
                            nx.draw(G_clean, pos, with_labels=True, node_color=node_colors, edge_color="gray", node_size=2500, font_size=8)
                            plt.title("Network Graph: Patient-Disease Associations")
                            plt.savefig("static/plot.png")
                            """
                            
                            " "
                        ),
                    }
                    
                ]
            )
    

        # Extract and clean generated Python code
        generated_code = chat_completion.choices[0].message.content
        match = re.search(r"python\s*(.?)\s", generated_code, re.DOTALL)
        return match.group(1).strip() if match else None

    except Exception as e:
        return f"Error generating visualization: {e}"


# =====================================
#  Query Handling Function
# =====================================
"""
This function determines the appropriate tool based on the query type:
  - If the query requests visualization, use text_to_aql_to_nxalgorithm
  - Otherwise, use text_to_aql_to_text
"""
def query_graph(query: str) -> str:
    return text_to_aql_to_nxalgorithm(query) if any(
        keyword in query.lower() for keyword in ["visualize", "show"]
    ) else text_to_aql_to_text(query)


# =====================================
#  Flask Routes
# =====================================
"""
This section defines the Flask routes for handling user requests:
  - /: Renders the homepage with an input form.
  - /query: Handles text-based queries and displays the result.
  - /visualize: Handles visualization queries and displays the plot.
"""

@app.route("/")
def home() -> str:
    """Renders the homepage with an input form."""
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def handle_query() -> str:
    """
    Handles text-based queries, processes them,
    and displays the result.
    """
    user_query = request.form["query"]
    result = text_to_aql_to_text(user_query)
    return render_template("index.html", result=result)


@app.route("/visualize", methods=["POST"])
def visualize_query() -> str:
    """
    Handles visualization queries, generates a graph,
    and displays the plot.
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
"""
This section runs the Flask application when the script is executed.
"""
if __name__ == "__main__":
    app.run(debug=False)

