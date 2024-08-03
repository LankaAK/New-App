import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the file (either CSV or Excel)
@st.cache
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file format")
        return None

st.title("Interactive Dashboard")

# File uploader for CSV or Excel file
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load data from the uploaded file
    df = load_data(uploaded_file)
    
    if df is not None:
        st.write("Data Preview")
        st.dataframe(df.head())  # Display a preview of the data

        # Sidebar for general and comparison options
        st.sidebar.title("Options")

        # General Bar Chart
        st.sidebar.subheader("General Bar Chart")
        general_column = st.sidebar.selectbox("Select column for general bar chart", df.columns.tolist())
        
        if general_column:
            st.write(f"General Bar Chart for {general_column}")
            counts = df[general_column].value_counts()
            fig, ax = plt.subplots()
            counts.plot(kind='bar', ax=ax)
            ax.set_xlabel(general_column)
            ax.set_ylabel('Count')
            ax.set_title(f'Bar Chart of {general_column}')
            st.pyplot(fig)

        # Comparison Bar Chart
        st.sidebar.subheader("Comparison Bar Chart")
        comparison_column = st.sidebar.selectbox("Select column for comparison", df.columns.tolist())
        
        if comparison_column:
            unique_values = df[comparison_column].unique()
            selected_values = st.sidebar.multiselect(f"Select values to compare in {comparison_column}", unique_values)
            
            if selected_values:
                filtered_df = df[df[comparison_column].isin(selected_values)]
                st.write(f"Filtered Data ({len(filtered_df)} records)")
                st.dataframe(filtered_df)
                
                # Create a bar chart for selected values
                st.write(f"Comparison Bar Chart for {comparison_column}")
                counts = filtered_df[comparison_column].value_counts()
                fig, ax = plt.subplots()
                counts.plot(kind='bar', ax=ax)
                ax.set_xlabel(comparison_column)
                ax.set_ylabel('Count')
                ax.set_title(f'Bar Chart of {comparison_column} for Selected Values')
                st.pyplot(fig)
            else:
                st.write("Please select values to compare")
import pandas as pd
import numpy as np
import streamlit as st
df = pd.DataFrame(np.random.randn(500, 2) / [50, 50] + [37.76, -122.4],columns=['lat', 'lon'])
st.map(df)
