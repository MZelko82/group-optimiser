import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from optimizer import find_optimal_allocation_n_groups, get_box_weights, plot_group_distributions

st.set_page_config(page_title="Group Optimizer", layout="wide")

st.title("Group Allocation Optimizer")
st.write("""
Upload your data file (CSV or Excel) and optimize group allocations based on a numeric column while keeping specified groups together.
The app uses Integer Linear Programming (ILP) to find the mathematically optimal solution that minimizes differences between groups.
Note: Larger datasets with many boxes and groups may take longer to process.
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

        # Group configuration
        st.write("### Group Configuration")
        st.write("""
        Specify the number of groups and their names. The optimizer will ensure that:
        1. Items marked to stay together (same box/cage) will be assigned to the same group
        2. Groups will have approximately equal sizes
        3. The total values (e.g., weights) will be as similar as possible across groups
        """)
        
        n_groups = st.number_input("Number of groups:", min_value=2, max_value=10, value=2)
        
        group_names = []
        cols = st.columns(min(n_groups, 4))  # Show up to 4 columns
        for i in range(n_groups):
            col_idx = i % 4
            with cols[col_idx]:
                group_name = st.text_input(f"Name for Group {i+1}:", value=f"Group {i+1}")
                group_names.append(group_name)

        # Check for duplicate group names
        if len(set(group_names)) != len(group_names):
            st.error("Please ensure all group names are unique!")
            st.stop()
        
        if st.button("Optimize Groups"):
            with st.spinner("Finding optimal allocation... This may take a moment for larger datasets."):
                try:
                    # Calculate box weights
                    st.write("### Processing Data")
                    st.write("1. Loading and preprocessing data...")
                    box_weights = get_box_weights(
                        df, 
                        value_col=value_column,
                        box_col=group_column,
                        strain_col=strain_column
                    )
                    
                    st.write("Box weights calculated:")
                    st.dataframe(box_weights)
                    
                    # Find optimal allocation
                    st.write("2. Finding optimal allocation...")
                    st.write("Constraints:")
                    total_subjects = len(df)
                    subjects_per_group = total_subjects // n_groups
                    remainder = total_subjects % n_groups
                    st.write(f"- Total subjects: {total_subjects}")
                    st.write(f"- Target subjects per group: {subjects_per_group} (±{1 if remainder > 0 else 0})")
                    st.write(f"- Boxes must stay together")
                    st.write(f"- Minimizing weight difference between groups")
                    
                    results = find_optimal_allocation_n_groups(
                        box_weights,
                        n_groups=n_groups,
                        group_names=group_names,
                        strain_col=strain_column
                    )
                    
                    # Create output DataFrame
                    st.write("3. Creating output...")
                    output_df = df.copy()
                    
                    # Prepare data for plotting
                    df_plot = df.copy()
                    df_plot['Rat Box'] = df_plot[group_column]
                    
                    if strain_column:
                        strains = df[strain_column].unique()
                    else:
                        strains = ['Group']
                    
                    st.write(f"Processing {len(strains)} strain(s): {strains}")
                        
                    # Assign groups in output DataFrame
                    for strain in strains:
                        st.write(f"Assigning groups for strain: {strain}")
                        for group_name, boxes in results[strain]['groups'].items():
                            st.write(f"- {group_name}: {len(boxes)} boxes")
                            mask = df_plot[group_column].isin(boxes)
                            if strain_column:
                                mask &= (df_plot[strain_column] == strain)
                            output_df.loc[mask, 'Allocated_Group'] = group_name
                    
                    # Verify all rows have been assigned
                    unassigned = output_df[output_df['Allocated_Group'].isna()]
                    if len(unassigned) > 0:
                        st.error("Error: Some items were not assigned to groups.")
                        st.write("### Unassigned Items")
                        st.dataframe(unassigned)
                        st.write("### Debug Information")
                        st.write("Please check that:")
                        st.write("1. All items have valid numeric values")
                        st.write(f"Current value column: {value_column}")
                        st.write(f"Value types: {df[value_column].dtype}")
                        st.write(f"Sample values: {df[value_column].head()}")
                        st.write("2. The number of items is divisible by the number of groups")
                        total_subjects = len(df)
                        st.write(f"Total subjects: {total_subjects}")
                        st.write(f"Number of groups: {n_groups}")
                        st.write(f"Subjects per group: {total_subjects // n_groups}")
                        st.write(f"Remainder: {total_subjects % n_groups}")
                        st.write("Box summary:")
                        box_summary = df.groupby(group_column).agg({
                            value_column: ['count', 'sum', 'mean'],
                        }).reset_index()
                        box_summary.columns = [group_column, 'Subjects', 'Total Weight', 'Mean Weight']
                        st.dataframe(box_summary)
                        st.stop()
                    
                    # Display results
                    st.write("### Results")
                    st.write("The optimizer has found the mathematically optimal allocation that minimizes weight differences between groups while keeping subjects in the same box together.")
                    
                    # Show group summary
                    st.write("### Group Summary")
                    group_summary = output_df.groupby('Allocated_Group').agg({
                        value_column: ['count', 'sum', 'mean'],
                        group_column: lambda x: ', '.join(sorted(set(x.astype(str))))
                    }).reset_index()
                    group_summary.columns = ['Group', 'Subjects', 'Total Weight', 'Mean Weight', 'Boxes']
                    st.dataframe(group_summary)
                    
                    # Show full allocation
                    st.write("### Full Allocation")
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
                        st.write("Group totals:")
                        for group_name, total in results[strain]['group_weights'].items():
                            st.write(f"- {group_name}: {total:.1f}")
                        st.write(f"Variance between groups: {results[strain]['variance']:.2f}")
                        st.write(f"Maximum difference between groups: {results[strain]['max_difference']:.2f}")
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
