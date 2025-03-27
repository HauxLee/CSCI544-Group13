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
    
@tool
def clean_data(
    data: Annotated[pd.DataFrame, "DataFrame to clean"],
    operations: Annotated[list, "List of cleaning operations to perform"],
    columns: Annotated[list, "Columns to apply operations to"] = None
) -> pd.DataFrame:
    """
    Clean and preprocess DataFrame data.
    
    Args:
        data: DataFrame to clean
        operations: List of operations to perform, can include:
                   ['drop_na', 'fill_na_mean', 'fill_na_median', 'fill_na_mode', 
                    'remove_outliers', 'normalize', 'standardize', 'encode_categorical']
        columns: Columns to apply operations to. If None, applies to all suitable columns.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        result_df = data.copy()
        
        if columns is None:
            # Determine suitable columns based on data types
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = result_df.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            # Verify all specified columns exist
            for col in columns:
                if col not in result_df.columns:
                    return f"Column '{col}' not found in DataFrame"
            
            numeric_cols = [col for col in columns if col in result_df.select_dtypes(include=[np.number]).columns]
            categorical_cols = [col for col in columns if col in result_df.select_dtypes(include=['object', 'category']).columns]
        
        # Apply specified operations
        for operation in operations:
            if operation == 'drop_na':
                if columns:
                    result_df = result_df.dropna(subset=columns)
                else:
                    result_df = result_df.dropna()
                    
            elif operation == 'fill_na_mean':
                for col in numeric_cols:
                    result_df[col] = result_df[col].fillna(result_df[col].mean())
                    
            elif operation == 'fill_na_median':
                for col in numeric_cols:
                    result_df[col] = result_df[col].fillna(result_df[col].median())
                    
            elif operation == 'fill_na_mode':
                for col in columns if columns else result_df.columns:
                    result_df[col] = result_df[col].fillna(result_df[col].mode()[0] if not result_df[col].mode().empty else None)
                    
            elif operation == 'remove_outliers':
                for col in numeric_cols:
                    Q1 = result_df[col].quantile(0.25)
                    Q3 = result_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    result_df = result_df[(result_df[col] >= lower_bound) & (result_df[col] <= upper_bound)]
                    
            elif operation == 'normalize':
                for col in numeric_cols:
                    min_val = result_df[col].min()
                    max_val = result_df[col].max()
                    if max_val > min_val:  # Prevent division by zero
                        result_df[col] = (result_df[col] - min_val) / (max_val - min_val)
                    
            elif operation == 'standardize':
                for col in numeric_cols:
                    mean = result_df[col].mean()
                    std = result_df[col].std()
                    if std > 0:  # Prevent division by zero
                        result_df[col] = (result_df[col] - mean) / std
                    
            elif operation == 'encode_categorical':
                for col in categorical_cols:
                    # Simple label encoding
                    result_df[f"{col}_encoded"] = pd.Categorical(result_df[col]).codes
            
            else:
                return f"Unsupported operation: {operation}"
        
        return result_df
    
    except Exception as e:
        return f"Failed to clean data. Error: {repr(e)}"
    
@tool
def group_and_aggregate(
    data: Annotated[pd.DataFrame, "DataFrame to analyze"],
    group_by: Annotated[list, "Columns to group by"],
    aggregations: Annotated[dict, "Dictionary mapping columns to aggregation functions"]
) -> pd.DataFrame:
    """
    Group DataFrame by specified columns and apply aggregation functions.
    
    Args:
        data: DataFrame to analyze
        group_by: Columns to group by
        aggregations: Dictionary mapping columns to aggregation functions
                      Example: {"sales": ["sum", "mean"], "quantity": "max"}
        
    Returns:
        pd.DataFrame: DataFrame with grouped and aggregated data
    """
    try:
        # Verify all columns exist
        all_cols = group_by + list(aggregations.keys())
        for col in all_cols:
            if col not in data.columns:
                return f"Column '{col}' not found in DataFrame"
        
        # Perform groupby and aggregation
        result = data.groupby(group_by).agg(aggregations).reset_index()
        
        # Flatten multi-level column names if needed
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = ['_'.join(col).strip('_') for col in result.columns.values]
            
        return result
    
    except Exception as e:
        return f"Failed to group and aggregate data. Error: {repr(e)}"
@tool
def time_series_analysis(
    data: Annotated[pd.DataFrame, "DataFrame containing time series data"],
    date_column: Annotated[str, "Column containing dates"],
    value_column: Annotated[str, "Column containing values to analyze"],
    frequency: Annotated[str, "Frequency for resampling (D=daily, W=weekly, M=monthly, etc.)"] = None,
    operations: Annotated[list, "List of operations to perform"] = ["trend", "seasonality", "moving_avg"]
) -> dict:
    """
    Perform time series analysis on DataFrame data.
    
    Args:
        data: DataFrame containing time series data
        date_column: Column containing dates
        value_column: Column containing values to analyze
        frequency: Frequency for resampling (D=daily, W=weekly, M=monthly, etc.)
        operations: List of operations to perform
        
    Returns:
        dict: Dictionary with analysis results
    """
    try:
        # Verify columns exist
        if date_column not in data.columns:
            return f"Date column '{date_column}' not found in DataFrame"
        if value_column not in data.columns:
            return f"Value column '{value_column}' not found in DataFrame"
        
        # Ensure date column is datetime
        df = data.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Set date as index
        df = df.set_index(date_column)
        
        # Resample if frequency is specified
        if frequency:
            df_resampled = df[value_column].resample(frequency).mean()
        else:
            df_resampled = df[value_column]
        
        results = {}
        
        # Calculate trend using rolling mean
        if "trend" in operations:
            window_size = max(len(df_resampled) // 10, 2)  # Use 10% of data points or at least 2
            results["trend"] = df_resampled.rolling(window=window_size).mean().dropna().tolist()
            results["trend_dates"] = df_resampled.rolling(window=window_size).mean().dropna().index.astype(str).tolist()
        
        # Calculate moving average
        if "moving_avg" in operations:
            short_window = max(len(df_resampled) // 20, 2)  # 5% of data points or at least 2
            long_window = max(len(df_resampled) // 10, 2)   # 10% of data points or at least 2
            
            results["short_moving_avg"] = df_resampled.rolling(window=short_window).mean().dropna().tolist()
            results["long_moving_avg"] = df_resampled.rolling(window=long_window).mean().dropna().tolist()
            results["moving_avg_dates"] = df_resampled.rolling(window=long_window).mean().dropna().index.astype(str).tolist()
        
        # Calculate basic statistics
        results["statistics"] = {
            "mean": float(df_resampled.mean()),
            "std": float(df_resampled.std()),
            "min": float(df_resampled.min()),
            "max": float(df_resampled.max()),
            "count": int(df_resampled.count())
        }
        
        # Calculate year-over-year or month-over-month growth if enough data
        if len(df_resampled) > 12 and "growth_rate" in operations:
            if frequency in ['M', 'MS', 'ME']:
                # Month-over-month
                results["monthly_growth"] = df_resampled.pct_change().dropna().tolist()
                results["monthly_growth_dates"] = df_resampled.pct_change().dropna().index.astype(str).tolist()
            elif frequency in ['Y', 'YS', 'YE', 'A']:
                # Year-over-year
                results["yearly_growth"] = df_resampled.pct_change().dropna().tolist()
                results["yearly_growth_dates"] = df_resampled.pct_change().dropna().index.astype(str).tolist()
        
        return results
    
    except Exception as e:
        return f"Failed to perform time series analysis. Error: {repr(e)}"
    
@tool
def merge_dataframes(
    left_df: Annotated[pd.DataFrame, "Left DataFrame"],
    right_df: Annotated[pd.DataFrame, "Right DataFrame"],
    how: Annotated[str, "Type of merge: 'inner', 'outer', 'left', or 'right'"] = "inner",
    on: Annotated[list, "Column(s) to join on in both DataFrames"] = None,
    left_on: Annotated[list, "Column(s) to join on in left DataFrame"] = None,
    right_on: Annotated[list, "Column(s) to join on in right DataFrame"] = None
) -> pd.DataFrame:
    """
    Merge two DataFrames based on common columns.
    
    Args:
        left_df: Left DataFrame
        right_df: Right DataFrame
        how: Type of merge ('inner', 'outer', 'left', 'right')
        on: Column(s) to join on in both DataFrames
        left_on: Column(s) to join on in left DataFrame
        right_on: Column(s) to join on in right DataFrame
        
    Returns:
        pd.DataFrame: Merged DataFrame
    """
    try:
        # Validate merge type
        valid_merge_types = ['inner', 'outer', 'left', 'right']
        if how not in valid_merge_types:
            return f"Invalid merge type. Must be one of {valid_merge_types}"
        
        # Perform merge
        if on is not None:
            # Validate 'on' columns exist in both DataFrames
            for col in on:
                if col not in left_df.columns:
                    return f"Column '{col}' not found in left DataFrame"
                if col not in right_df.columns:
                    return f"Column '{col}' not found in right DataFrame"
            
            result = pd.merge(left_df, right_df, how=how, on=on)
            
        elif left_on is not None and right_on is not None:
            # Validate left_on columns exist in left DataFrame
            for col in left_on:
                if col not in left_df.columns:
                    return f"Column '{col}' not found in left DataFrame"
            
            # Validate right_on columns exist in right DataFrame
            for col in right_on:
                if col not in right_df.columns:
                    return f"Column '{col}' not found in right DataFrame"
            
            result = pd.merge(left_df, right_df, how=how, left_on=left_on, right_on=right_on)
            
        else:
            return "Must specify either 'on' or both 'left_on' and 'right_on'"
        
        return result
    
    except Exception as e:
        return f"Failed to merge DataFrames. Error: {repr(e)}"

    
