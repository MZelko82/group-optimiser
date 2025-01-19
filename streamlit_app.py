import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from optimizer import find_optimal_allocation, get_box_weights, plot_group_distributions

st.set_page_config(page_title="Group Optimizer", layout="wide")

st.title("Group Allocation Optimizer")
st.write("""
Upload your data file (CSV or Excel) and optimize group allocations based on a numeric column while keeping specified groups together.
""")

# File upload
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    # Load the data
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.write("### Data Preview")
        st.dataframe(df.head())
        
        # Column selection
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            st.error("No numeric columns found in the data!")
            st.stop()
            
        col1, col2 = st.columns(2)
        
        with col1:
            value_column = st.selectbox(
                "Select the column to optimize groups on (numeric values):",
                numeric_columns
            )
        
        with col2:
            group_column = st.selectbox(
                "Select the column that identifies which items must stay together:",
                df.columns
            )
        
        strain_column = None
        if len(df.columns) > 2:  # If there are more columns, allow strain selection
            strain_column = st.selectbox(
                "Optional: Select a column to separate optimizations by (e.g., strain/type):",
                ['None'] + list(df.columns)
            )
            if strain_column == 'None':
                strain_column = None
        
        if st.button("Optimize Groups"):
            # Calculate box weights
            box_weights = get_box_weights(
                df, 
                value_col=value_column,
                box_col=group_column,
                strain_col=strain_column
            )
            
            # Find optimal allocation
            results = find_optimal_allocation(box_weights, strain_col=strain_column)
            
            # Create output DataFrame
            output_df = df.copy()
            output_df['Allocated_Group'] = 'Unassigned'
            
            # Prepare data for plotting
            df_plot = df.copy()
            df_plot['Rat Box'] = df_plot[group_column]
            
            if strain_column:
                strains = df[strain_column].unique()
            else:
                strains = ['Group']
                
            for strain in strains:
                # Assign CON group
                mask = df_plot[group_column].isin(results[strain]['CON'])
                if strain_column:
                    mask &= (df_plot[strain_column] == strain)
                output_df.loc[mask, 'Allocated_Group'] = 'CON'
                
                # Assign CR group
                mask = df_plot[group_column].isin(results[strain]['CR'])
                if strain_column:
                    mask &= (df_plot[strain_column] == strain)
                output_df.loc[mask, 'Allocated_Group'] = 'CR'
            
            # Display results
            st.write("### Results")
            st.dataframe(output_df)
            
            # Create plots
            fig = plot_group_distributions(
                df_plot, 
                results, 
                value_col=value_column,
                strain_col=strain_column
            )
            st.pyplot(fig)
            
            # Provide download link
            output = BytesIO()
            output_df.to_csv(output, index=False)
            output.seek(0)
            st.download_button(
                label="Download Results CSV",
                data=output,
                file_name="optimized_groups.csv",
                mime="text/csv"
            )
            
            # Display statistics
            st.write("### Group Statistics")
            for strain in strains:
                st.write(f"\n**{strain}**")
                st.write(f"- CON group total: {results[strain]['CON_weight']:.1f}")
                st.write(f"- CR group total: {results[strain]['CR_weight']:.1f}")
                st.write(f"- Difference: {results[strain]['weight_difference']:.1f}")
                st.write(f"- Percent difference: {(results[strain]['weight_difference'] / results[strain]['CON_weight'] * 100):.1f}%")
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
