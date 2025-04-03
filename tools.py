import os
import json
import re
from typing import Annotated
import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns
from fpdf import FPDF
import matplotlib
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import matplotlib
import matplotlib.pyplot as plt
from langchain_experimental.utilities import PythonREPL
import math
from scipy.stats import chi2_contingency
matplotlib.use('Agg')



"""
tools.py

- load_csv_file: Loads a CSV file and returns it as a Pandas DataFrame.
- get_user_input_tool: Captures user input dynamically during an interaction and formats it as a dictionary to be added to the Agent's conversation.
"""

# ======================================
#
# - load_csv_file: Loads a CSV file and returns it as a Pandas DataFrame.
#   Input: file_path (str), Output: Pandas DataFrame (pd.DataFrame).
#
# - get_user_input_tool: Captures user input dynamically during an interaction and formats it as a dictionary to be added to the Agent's conversation.  
#   Input: None. Output: User input message (dict) or error message (str).
# 
# ======================================
    
@tool
def load_csv_file(
    file_path: Annotated[str, "The path to the CSV file to load."]
) -> pd.DataFrame:
    """
    Load a CSV file and return it as a Pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file to load.

    Returns:
        pd.DataFrame: The loaded data as a Pandas DataFrame, or an error message if the process failed.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        return df
    
    except Exception as e:
        return f"Failed to load the CSV file. Error: {repr(e)}"


# @tool
# def get_user_input_tool() -> dict:
#     """
#     Get user input dynamically during an ongoing interaction with the Agent.

#     Returns:
#         dict: A dictionary representing the user's input message to be added to the Agent's conversation.
#     """
#     try:
#         user_input_content = input("(Send a message to the Agent): ")

#         combined_message = f"{user_input_content}."

#         user_message = {
#             "messages": [
#                 HumanMessage(
#                     content=combined_message
#                 )
#             ],
#             "sender": "Human"
#         }

#         return user_message
    
#     except Exception as e:
#         return {
#             "error": f"Failed to get user input. Error: {repr(e)}"
#         }
@tool
def get_user_input_tool(
    prompt: Annotated[str, "Optional prompt to show the user"] = None
) -> dict:
    """
    Get user input dynamically during an ongoing interaction with the Agent.

    Args:
        prompt (str, optional): Custom prompt to show the user. Defaults to None.

    Returns:
        dict: A dictionary representing the user's input message to be added to the Agent's conversation.
    """
    try:
        # 使用自定义提示或默认提示
        display_prompt = prompt if prompt else "(Send a message to the Agent): "
        user_input_content = input(display_prompt)

        combined_message = f"{user_input_content}"

        user_message = {
            "messages": [
                HumanMessage(
                    content=combined_message
                )
            ],
            "sender": "Human"
        }

        return user_message
    
    except Exception as e:
        return {
            "error": f"Failed to get user input. Error: {repr(e)}"
        }
@tool
def execute_sql_query(
    query: Annotated[str, "The SQL query to execute."],
    db_name: Annotated[str, "Database name, default is the configured database."] = "default"
) -> list:
    """
    Execute a SQL query and return the results.
    
    Args:
        query (str): The SQL query to execute.
        db_name (str, optional): The name of the database to query.
        
    Returns:
        list: The query results as a list of dictionaries.
    """
    try:
        from database import DatabaseManager
        
        # Create database manager instance
        db_manager = DatabaseManager()
        
        # Execute the query
        results = db_manager.execute_query(query)
        
        # Convert SQLite Row objects to dictionaries
        dict_results = []
        for row in results:
            # Convert each row to a dictionary
            dict_row = {key: row[key] for key in row.keys()} if hasattr(row, 'keys') else dict(row)
            dict_results.append(dict_row)
        
        # Close connection
        db_manager.disconnect()
        
        return dict_results
    except Exception as e:
        return f"Failed to execute SQL query. Error: {repr(e)}"


@tool
def get_database_schema(
    db_name: Annotated[str, "Database name, defaults to the configured database."] = "default"
) -> dict:
    """
    Extract and return the database schema including tables, columns, and relationships.
    
    Args:
        db_name (str, optional): The name of the database. Defaults to "default".
        
    Returns:
        dict: A dictionary containing the database schema structure, or an error message if extraction failed.
    """
    try:
        from database import DatabaseManager
        
        # Create database manager instance
        db_manager = DatabaseManager()
        
        schema = {}
        
        # For SQLite, we can query the sqlite_master table
        if db_manager.db_type == 'sqlite':
            # Get all tables
            tables = db_manager.execute_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            
            schema['tables'] = {}
            
            # Get columns for each table
            for table in tables:
                table_name = table['name']
                columns = db_manager.execute_query(f"PRAGMA table_info({table_name})")
                
                # Format column info
                schema['tables'][table_name] = {
                    'columns': [
                        {
                            'name': col['name'],
                            'type': col['type'],
                            'primary_key': bool(col['pk']),
                            'nullable': not bool(col['notnull']),
                            'default': col['dflt_value']
                        }
                        for col in columns
                    ]
                }
                
                # Get foreign keys
                foreign_keys = db_manager.execute_query(f"PRAGMA foreign_key_list({table_name})")
                if foreign_keys:
                    schema['tables'][table_name]['foreign_keys'] = [
                        {
                            'column': fk['from'],
                            'references_table': fk['table'],
                            'references_column': fk['to']
                        }
                        for fk in foreign_keys
                    ]
        
        # Close connection
        db_manager.disconnect()
        
        return schema
    except Exception as e:
        return f"Failed to extract database schema. Error: {repr(e)}"

# @tool
# def generate_data_visualization(
#     query_result: Annotated[str, "SQL query result or path to CSV file to visualize."],
#     chart_type: Annotated[str, "Type of chart to generate (bar, line, pie, scatter)."],
#     x_column: Annotated[str, "Column name for x-axis or categories."],
#     y_column: Annotated[str, "Column name for y-axis or values."] = None,
#     title: Annotated[str, "Chart title."] = "Data Visualization",
#     output_path: Annotated[str, "Path to save the chart image."] = "generated_files/chart.png"
# ) -> str:
#     """
#     Generate a visualization from query results or CSV data and save it as an image.
    
#     Args:
#         query_result (str): SQL query result (as JSON string) or path to CSV file to visualize.
#         chart_type (str): Type of chart to generate (bar, line, pie, scatter).
#         x_column (str): Column name for x-axis or categories.
#         y_column (str, optional): Column name for y-axis or values.
#         title (str, optional): Chart title. Defaults to "Data Visualization".
#         output_path (str, optional): Path to save the chart image. Defaults to "generated_files/chart.png".
        
#     Returns:
#         str: Path to the generated chart image, or an error message if generation failed.
#     """
#     try:
#         # Ensure output directory exists
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
#         # Convert query result to DataFrame
#         if query_result.endswith('.csv'):
#             # If it's a CSV file path
#             data = pd.read_csv(query_result)
#         else:
#             # Assume it's a JSON string of query results
#             try:
#                 import json
#                 result_list = json.loads(query_result)
#                 data = pd.DataFrame(result_list)
#             except:
#                 return "Unable to parse query result. Please provide valid JSON or a CSV file path."
        
#         # Generate visualization from the DataFrame
#         plt.figure(figsize=(10, 6))
        
#         if chart_type.lower() == 'bar':
#             plt.bar(data[x_column], data[y_column])
#             plt.xlabel(x_column)
#             plt.ylabel(y_column)
        
#         elif chart_type.lower() == 'line':
#             plt.plot(data[x_column], data[y_column], marker='o')
#             plt.xlabel(x_column)
#             plt.ylabel(y_column)
        
#         elif chart_type.lower() == 'pie':
#             plt.pie(data[y_column], labels=data[x_column], autopct='%1.1f%%')
        
#         elif chart_type.lower() == 'scatter':
#             plt.scatter(data[x_column], data[y_column])
#             plt.xlabel(x_column)
#             plt.ylabel(y_column)
        
#         else:
#             return f"Unsupported chart type: {chart_type}. Use bar, line, pie, or scatter."
        
#         plt.title(title)
#         plt.tight_layout()
        
#         # Save the figure
#         plt.savefig(output_path)
#         plt.close()
        
#         return f"Chart generated and saved to {output_path}"
    
#     except Exception as e:
#         return f"Failed to generate visualization. Error: {repr(e)}"
