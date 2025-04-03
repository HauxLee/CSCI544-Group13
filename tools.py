import os
import json
import re
from typing import Annotated, Optional, Dict, Any
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
import sqlglot
from sqlglot.errors import ParseError
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
        dict: A dictionary representing the user's input message to be added to the Agent's conversation,
             or a dictionary with an exit flag if the user wants to exit.
    """
    try:
        display_prompt = prompt if prompt else "\n👤 You: "
        user_input_content = input(display_prompt).strip().lower()

        # Check for exit commands
        if user_input_content in ['exit', 'quit']:
            return {
                "exit": True,
                "message": "Goodbye! Thanks for using Database Assistant."
            }

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
def validate_sql_query(
    query: Annotated[str, "The SQL query to validate."],
    dialect: Annotated[str, "SQL dialect to use (e.g., 'sqlite', 'mysql', 'postgresql')"] = "sqlite"
) -> dict:
    """
    Validates a SQL query using SQLGlot parser and optional schema validation.
    
    Args:
        query (str): The SQL query to validate
        dialect (str): SQL dialect to use for parsing (default: 'sqlite')
        
    Returns:
        dict: Validation results containing status, messages, and parsed information
    """
    def parse_sql(sql: str) -> tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Parse SQL and return validation status"""
        try:
            # Parse the SQL query
            parsed = sqlglot.parse_one(sql, read=dialect)
            
            # Extract useful information from the parsed query
            tables = parsed.find_all(sqlglot.exp.Table)
            columns = parsed.find_all(sqlglot.exp.Column)
            
            # Get query type (SELECT, INSERT, etc.)
            query_type = parsed.key
            
            return True, None, {
                "query_type": query_type,
                "tables": [str(t) for t in tables],
                "columns": [str(c) for c in columns],
                "normalized_sql": parsed.sql(dialect=dialect)
            }
            
        except ParseError as e:
            return False, f"SQL syntax error: {str(e)}", None
        except Exception as e:
            return False, f"Validation error: {str(e)}", None

    def validate_against_schema(parsed_info: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate parsed SQL against database schema"""
        try:
            from database import DatabaseManager
            
            db_manager = DatabaseManager()
            schema = db_manager.get_table_names()
            
            # Check if all referenced tables exist
            for table in parsed_info["tables"]:
                if table.upper() not in [t.upper() for t in schema]:
                    return False, f"Table '{table}' does not exist in the database"
            
            return True, None
            
        except Exception as e:
            return False, f"Schema validation error: {str(e)}"

    try:
        # Step 1: Parse and validate SQL syntax
        is_valid, error_msg, parsed_info = parse_sql(query)
        if not is_valid:
            return {
                "is_valid": False,
                "message": error_msg,
                "validation_step": "syntax",
                "details": None
            }
        
        # Step 2: Validate against schema if parsing succeeded
        schema_valid, schema_error = validate_against_schema(parsed_info)
        if not schema_valid:
            return {
                "is_valid": False,
                "message": schema_error,
                "validation_step": "schema",
                "details": parsed_info
            }
        
        # All validations passed
        return {
            "is_valid": True,
            "message": "Query validation successful",
            "validation_step": "complete",
            "details": parsed_info
        }
        
    except Exception as e:
        return {
            "is_valid": False,
            "message": f"Unexpected error during validation: {str(e)}",
            "validation_step": "error",
            "details": None
        }

@tool
def execute_sql_query(
    query: Annotated[str, "The SQL query to execute."],
    db_name: Annotated[str, "Database name, default is the configured database."] = "default",
    max_retries: Annotated[int, "Maximum number of retry attempts"] = 3
) -> list:
    """
    Execute a SQL query with validation and retry logic.

    Args:
        query (str): The SQL query to execute.
        db_name (str, optional): The name of the database to query.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.

    Returns:
        list: The query results as a list of dictionaries.
    """
    attempt = 0
    last_error = None
    
    while attempt < max_retries:
        try:
            # First validate the query
            validation_result = validate_sql_query.invoke({"query": query})
            if not validation_result["is_valid"]:
                return f"Query validation failed: {validation_result['message']}"
            
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
            last_error = str(e)
            attempt += 1
            if attempt < max_retries:
                continue
            
    return f"Failed to execute SQL query after {max_retries} attempts. Last error: {last_error}"


@tool
def get_database_schema(
    db_name: Annotated[str, "Database name, defaults to the configured database."] = "default"
) -> dict:
    """
    Extract and return the database schema of the configured database.

    This tool introspects the database and returns a structured description of its tables,
    including column names, data types, primary key status, nullability, default values,
    and any foreign key relationships.

    Args:
        db_name (str, optional): The name of the database. Defaults to "default".

    Returns:
        dict: A structured dictionary describing the tables and their columns.
    """
    try:
        from database import DatabaseManager

        db_manager = DatabaseManager()

        if not hasattr(db_manager, "get_table_names") or not hasattr(db_manager, "get_table_schema"):
            return {"error": "Schema introspection not supported for this database type."}

        schema = {
            "db_type": db_manager.db_type,
            "tables": {}
        }

        for table_name in db_manager.get_table_names():
            schema["tables"][table_name] = {
                "columns": db_manager.get_table_schema(table_name)
            }

        db_manager.disconnect()
        return schema

    except Exception as e:
        return {"error": f"Failed to extract database schema: {repr(e)}"}


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
