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

        # Sidebar for filtering options
        st.sidebar.title("Filter Options")
        
        # Display unique values for a column for filtering
        columns = df.columns.tolist()
        selected_column = st.sidebar.selectbox("Select column to filter", columns)
        
        if selected_column:
            unique_values = df[selected_column].unique()
            selected_values = st.sidebar.multiselect(f"Select values for {selected_column}", unique_values)

            if selected_values:
                filtered_df = df[df[selected_column].isin(selected_values)]
                st.write(f"Filtered Data ({len(filtered_df)} records)")
                st.dataframe(filtered_df)
                
                # Create a bar chart for selected values
                st.write("Bar Chart of Selected Values")
                
                # Aggregate data for selected values
                counts = filtered_df[selected_column].value_counts()

                # Plotting
                fig, ax = plt.subplots()
                counts.plot(kind='bar', ax=ax)
                ax.set_xlabel(selected_column)
                ax.set_ylabel('Count')
                ax.set_title('Bar Chart of Selected Values')
                
                st.pyplot(fig)
            else:
                st.write("Please select values to filter")

        # Additional options
        st.sidebar.title("Additional Options")
        sort_column = st.sidebar.selectbox("Select column to sort", columns)
        sort_order = st.sidebar.radio("Sort order", ["Ascending", "Descending"])

        if sort_column:
            if sort_order == "Ascending":
                sorted_df = df.sort_values(by=sort_column)
            else:
                sorted_df = df.sort_values(by=sort_column, ascending=False)
            
            st.write("Sorted Data")
            st.dataframe(sorted_df)
            
            # Histogram for a numerical column
            numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
            if numerical_columns:
                selected_numerical_column = st.sidebar.selectbox("Select numerical column for histogram", numerical_columns)
                if selected_numerical_column:
                    st.write(f"Histogram of {selected_numerical_column}")
                    fig, ax = plt.subplots()
                    df[selected_numerical_column].hist(ax=ax)
                    ax.set_xlabel(selected_numerical_column)
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Histogram of {selected_numerical_column}')
                    st.pyplot(fig)
            else:
                st.write("No numerical columns available for histogram.")
