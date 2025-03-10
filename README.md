# README.md

## Project Overview

This project is a Flask-based web application that integrates with ArangoDB, LangChain, and GROQ to provide a powerful interface for querying and visualizing graph data. The application allows users to input natural language queries, which are then translated into ArangoDB Query Language (AQL) using LangChain. The results can be visualized using NetworkX and Matplotlib, and the visualizations are rendered in the web interface.

## Features

- **Natural Language to AQL Translation**: Convert natural language queries into AQL using LangChain and execute them on an ArangoDB database.
- **Graph Visualization**: Generate and visualize graph data using NetworkX and Matplotlib. The visualizations are saved as static images and displayed in the web interface.
- **GROQ Integration**: Utilize GROQ's language models for enhanced query processing and visualization code generation.
- **Flask Web Interface**: A simple web interface to interact with the application, submit queries, and view results.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.8 or higher
- ArangoDB (running locally or accessible via a network)
- GROQ API key

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables**:
   Create a `.env` file in the root directory and add your GROQ API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Configure ArangoDB**:
   Ensure ArangoDB is running and accessible. Update the connection details in the code if necessary:
   ```python
   clien = ArangoClient(hosts='http://localhost:8529')
   db = clien.db('_system', username='root', password='')
   ```

## Running the Application

1. **Start the Flask Application**:
   ```bash
   python app.py
   ```

2. **Access the Web Interface**:
   Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Usage

### Querying the Database

1. **Enter a Query**:
   - On the home page, enter a natural language query in the input box.
   - Click "Submit" to execute the query.

2. **View Results**:
   - The results of the query will be displayed on the same page.
   - If the query includes visualization keywords (e.g., "visualize", "show"), a graph will be generated and displayed.

### Example Queries

- **Simple Query**:
 
  List all users and their purchased products.
  

- **Visualization Query**:
  
  Visualize the network of users and their purchased products.
  

## Code Structure

- **app.py**: The main Flask application file containing routes, query handling, and visualization logic.
- **templates/index.html**: The HTML template for the web interface.
- **static/**: Directory for storing static files like generated plot images.

## Tools and Libraries

- **ArangoDB**: A multi-model database used for storing and querying graph data.
- **LangChain**: A framework for developing applications powered by language models, used here for translating natural language queries into AQL.
- **GROQ**: A language model API used for generating visualization code and enhancing query processing.
- **NetworkX**: A Python library for creating, manipulating, and studying complex networks.
- **Matplotlib**: A plotting library for creating static, animated, and interactive visualizations in Python.
- **Flask**: A lightweight web framework for building web applications in Python.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- **ArangoDB** for providing a powerful multi-model database.
- **LangChain** for enabling natural language to AQL translation.
- **GROQ** for their advanced language models.
- **NetworkX** and **Matplotlib** for graph visualization capabilities.
- **Flask** for making web application development simple and efficient.

## Contact

For any questions or feedback, please contact:

- **Pavan Kothapalli**: [pavan.kothapalli@bluepal.com](mail to:pavan.kothapalli@bluepal.com)  
- **Rithwik Thirumalaraju**: [rithwik.thirumalaraju@bluepal.com](mail to:rithwik.thirumalaraju@bluepal.com)  
- **Keerthi Datla**: [keerthi.datla@bluepal.com](mail to:keerthi.datla@bluepal.com)  

We welcome your input and are happy to assist with any inquiries or issues you may have!
