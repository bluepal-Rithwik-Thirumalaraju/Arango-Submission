from arango import ArangoClient
from langchain_community.graphs import ArangoGraph
from langchain.chains import ArangoGraphQAChain
from langchain_groq import ChatGroq
import re
from langchain_core.tools import tool
import networkx as nx
import json 
import os
import matplotlib.pyplot as plt  # For visualization
import textwrap
import plotly
from groq import Groq
from flask import Flask, render_template, request, jsonify
# Initialize Flask app
app = Flask(__name__)

# Set GROQ API key
os.environ["GROQ_API_KEY"] = "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"

# Initialize the ArangoDB client and connect to the database
clien = ArangoClient(hosts='http://localhost:8529')
db = clien.db('_system', username='root', password='')

# Instantiate the ArangoDB-LangChain Graph
arango_graph = ArangoGraph(db)

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),  # This is the default and can be omitted
)

# Tool 1: Translate text to AQL and back to text
@tool
def text_to_aql_to_text(query: str):
    """This tool is available to invoke the
    ArangoGraphQAChain object, which enables you to
    translate a Natural Language Query into AQL, execute
    the query, and translate the result back into Natural Language.
    """
    llm = ChatGroq(temperature=0, model_name="qwen-2.5-32b")
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=arango_graph,
        verbose=True,
        allow_dangerous_requests=True
    )
    result = chain.invoke(query)
    return str(result["result"])

# Tool 2: Translate text to AQL, build a NetworkX graph, run an algorithm, and visualize
@tool
def text_to_aql_to_nxalgorithm(query: str):
    """This tool takes a natural language query, generates a sample graph (nodes and edges)
    based on the query context, builds a NetworkX graph, runs an algorithm, and returns
    the raw result (FINAL_RESULT).
    """
    
    graph_prompt = f"""This tool is available to invoke the
    ArangoGraphQAChain object, which enables you to
    translate a Natural Language Query into AQL, execute
    the query : {query}, and translate the result back into Natural Language.
    """
    llm = ChatGroq(temperature=0, model_name="qwen-2.5-32b")
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=arango_graph,
        verbose=True,
        allow_dangerous_requests=True
    )
    result = chain.invoke(query)
    result = str(result["result"])


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
    

    
    text_to_nx = chat_completion.choices[0].message.content
    pattern = r"python\s*(.*?)\s*"
    match = re.search(pattern, text_to_nx, re.DOTALL)

    if match:
        text_to_nx_cleaned = match.group(1).strip()
        print(text_to_nx_cleaned)  # Output only the extracted code
    else:
        print("No match found")

    return text_to_nx_cleaned

# Define tools list
tools = [text_to_aql_to_text, text_to_aql_to_nxalgorithm]

# Query handler function
def query_graph(query):
    """Handles user queries and selects the appropriate tool."""
    if "visualize" in query.lower() or "show" in query.lower():
        tool = text_to_aql_to_nxalgorithm

    else:
        tool = text_to_aql_to_text

    result = tool.invoke(query)
    return result

# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def handle_query():
    user_query = request.form["query"]
    tool = text_to_aql_to_text
    result=tool.invoke(user_query)
    return render_template("index.html",result=result)

@app.route("/visualize",methods=["POST"])
def visualize_query():
    u_query = request.form["query"]
    tool = text_to_aql_to_nxalgorithm
    result=tool.invoke(u_query)
    exec(textwrap.dedent(result))
    return render_template("index.html",plot="static/plot.png")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=False)
