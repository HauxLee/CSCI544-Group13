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


@tool
def get_user_input_tool() -> dict:
    """
    Get user input dynamically during an ongoing interaction with the Agent.

    Returns:
        dict: A dictionary representing the user's input message to be added to the Agent's conversation.
    """
    try:
        user_input_content = input("(Send a message to the Agent): ")

        combined_message = f"{user_input_content}."

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
def create_visualization(
    data: Annotated[pd.DataFrame, "DataFrame containing the data to visualize"],
    x_column: Annotated[str, "Column name for x-axis"],
    y_column: Annotated[str, "Column name for y-axis"],
    plot_type: Annotated[str, "Type of plot (bar, line, scatter, histogram, boxplot)"] = "bar",
    title: Annotated[str, "Plot title"] = "",
    output_path: Annotated[str, "Path to save the visualization"] = "visualization.png"
) -> str:
    """
    Create a visualization from DataFrame data and save it to a file.
    
    Args:
        data: DataFrame containing the data to visualize
        x_column: Column name for x-axis
        y_column: Column name for y-axis (not needed for histogram)
        plot_type: Type of plot (bar, line, scatter, histogram, boxplot)
        title: Plot title
        output_path: Path to save the visualization
        
    Returns:
        str: Path to the saved visualization or error message
    """
    try:
        plt.figure(figsize=(10, 6))
        
        if plot_type == "bar":
            sns.barplot(x=x_column, y=y_column, data=data)
        elif plot_type == "line":
            sns.lineplot(x=x_column, y=y_column, data=data)
        elif plot_type == "scatter":
            sns.scatterplot(x=x_column, y=y_column, data=data)
        elif plot_type == "histogram":
            sns.histplot(data[x_column], kde=True)
        elif plot_type == "boxplot":
            sns.boxplot(x=x_column, y=y_column, data=data)
        else:
            return f"Unsupported plot type: {plot_type}"
        
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return f"Visualization saved to {output_path}"
    
    except Exception as e:
        return f"Failed to create visualization. Error: {repr(e)}"


@tool
def descriptive_statistics(
    data: Annotated[pd.DataFrame, "DataFrame to analyze"],
    columns: Annotated[list, "List of columns to analyze"] = None
) -> dict:
    """
    Generate descriptive statistics for specified columns in a DataFrame.
    
    Args:
        data: DataFrame to analyze
        columns: List of columns to analyze. If None, analyzes all numeric columns.
        
    Returns:
        dict: Dictionary with descriptive statistics for each column
    """
    try:
        if columns is None:
            # Select only numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            columns = numeric_data.columns.tolist()
        else:
            # Verify all specified columns exist
            for col in columns:
                if col not in data.columns:
                    return f"Column '{col}' not found in DataFrame"
            
            # Filter to only include specified columns
            numeric_data = data[columns].select_dtypes(include=[np.number])
        
        # Calculate statistics
        result = {}
        for col in numeric_data.columns:
            col_data = data[col].dropna()
            result[col] = {
                "count": len(col_data),
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "25%": float(col_data.quantile(0.25)),
                "75%": float(col_data.quantile(0.75))
            }
            
        return result
    
    except Exception as e:
        return f"Failed to generate descriptive statistics. Error: {repr(e)}"


@tool
def correlation_analysis(
    data: Annotated[pd.DataFrame, "DataFrame to analyze"],
    columns: Annotated[list, "List of columns to analyze correlation"] = None,
    method: Annotated[str, "Correlation method: 'pearson', 'spearman', or 'kendall'"] = "pearson"
) -> dict:
    """
    Calculate correlation coefficients between numeric columns in a DataFrame.
    
    Args:
        data: DataFrame to analyze
        columns: List of columns to analyze. If None, analyzes all numeric columns.
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        
    Returns:
        dict: Dictionary with correlation matrix and p-values
    """
    try:
        if columns is None:
            # Select only numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            columns = numeric_data.columns.tolist()
        else:
            # Verify all specified columns exist
            for col in columns:
                if col not in data.columns:
                    return f"Column '{col}' not found in DataFrame"
            
            # Filter to only include specified columns
            numeric_data = data[columns].select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return "Need at least two numeric columns to calculate correlation"
        
        # Calculate correlation
        corr_matrix = numeric_data.corr(method=method).round(3).to_dict()
        
        # Calculate p-values for Pearson correlation
        p_values = {}
        if method == "pearson":
            for col1 in numeric_data.columns:
                p_values[col1] = {}
                for col2 in numeric_data.columns:
                    if col1 != col2:
                        corr, p = stats.pearsonr(
                            numeric_data[col1].dropna(), 
                            numeric_data[col2].dropna()
                        )
                        p_values[col1][col2] = round(float(p), 4)
                    else:
                        p_values[col1][col2] = 0.0
        
        return {
            "correlation_matrix": corr_matrix,
            "p_values": p_values if method == "pearson" else "P-values only available for Pearson correlation"
        }
    
    except Exception as e:
        return f"Failed to perform correlation analysis. Error: {repr(e)}"


@tool
def export_to_pdf(
    content: Annotated[dict, "Dictionary with content to include in the PDF"],
    title: Annotated[str, "PDF title"] = "Data Analysis Report",
    output_path: Annotated[str, "Path to save the PDF"] = "report.pdf"
) -> str:
    """
    Export analysis results and visualizations to a PDF report.
    
    Args:
        content: Dictionary with content to include in the PDF
                Format: {"text": [list of text blocks], "images": [list of image paths]}
        title: PDF title
        output_path: Path to save the PDF
        
    Returns:
        str: Path to the saved PDF or error message
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Add title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, title, ln=True, align="C")
        pdf.ln(10)
        
        # Add text content
        if "text" in content:
            pdf.set_font("Arial", "", 12)
            for text in content["text"]:
                pdf.multi_cell(0, 10, text)
                pdf.ln(5)
        
        # Add images
        if "images" in content:
            for img_path in content["images"]:
                if os.path.exists(img_path):
                    pdf.add_page()
                    pdf.image(img_path, x=10, w=190)
                else:
                    pdf.multi_cell(0, 10, f"Image not found: {img_path}")
        
        # Save PDF
        pdf.output(output_path)
        return f"PDF report saved to {output_path}"
    
    except Exception as e:
        return f"Failed to export PDF. Error: {repr(e)}"


@tool
def statistical_test(
    data: Annotated[pd.DataFrame, "DataFrame containing the data"],
    test_type: Annotated[str, "Type of statistical test (t_test, chi_square, anova)"],
    columns: Annotated[list, "Columns to use for the test"],
    group_column: Annotated[str, "Column to use for grouping (for t-test and ANOVA)"] = None,
    alpha: Annotated[float, "Significance level"] = 0.05
) -> dict:
    """
    Perform statistical tests on DataFrame data.
    
    Args:
        data: DataFrame containing the data
        test_type: Type of statistical test (t_test, chi_square, anova)
        columns: Columns to use for the test
        group_column: Column to use for grouping (for t-test and ANOVA)
        alpha: Significance level
        
    Returns:
        dict: Dictionary with test results
    """
    try:
        result = {"test_type": test_type, "alpha": alpha}
        
        if test_type == "t_test":
            if group_column is None or len(columns) != 1:
                return "T-test requires one data column and one group column"
            
            column = columns[0]
            groups = data[group_column].unique()
            
            if len(groups) != 2:
                return "T-test requires exactly two groups"
            
            group1_data = data[data[group_column] == groups[0]][column].dropna()
            group2_data = data[data[group_column] == groups[1]][column].dropna()
            
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
            
            result.update({
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "group1": str(groups[0]),
                "group2": str(groups[1]),
                "significant": p_value < alpha
            })
            
        elif test_type == "chi_square":
            if len(columns) != 2:
                return "Chi-square test requires exactly two categorical columns"
            
            contingency_table = pd.crosstab(data[columns[0]], data[columns[1]])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            
            result.update({
                "chi2_statistic": float(chi2),
                "p_value": float(p),
                "degrees_of_freedom": int(dof),
                "significant": p < alpha,
                "contingency_table": contingency_table.to_dict()
            })
            
        elif test_type == "anova":
            if group_column is None or len(columns) != 1:
                return "ANOVA requires one data column and one group column"
            
            column = columns[0]
            groups = {}
            
            for group in data[group_column].unique():
                groups[group] = data[data[group_column] == group][column].dropna()
            
            if len(groups) < 2:
                return "ANOVA requires at least two groups"
            
            anova_results = stats.f_oneway(*list(groups.values()))
            
            result.update({
                "f_statistic": float(anova_results[0]),
                "p_value": float(anova_results[1]),
                "significant": anova_results[1] < alpha,
                "groups": list(groups.keys())
            })
            
        else:
            return f"Unsupported test type: {test_type}"
        
        return result
    
    except Exception as e:
        return f"Failed to perform statistical test. Error: {repr(e)}"

    
