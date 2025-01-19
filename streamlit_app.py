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
                    # Debug: Show input data
                    st.write("### Input Data")
                    st.write(f"DataFrame shape: {df.shape}")
                    st.write("First few rows:")
                    st.write(df.head())
                    st.write(f"\nColumns: {df.columns.tolist()}")
                    
                    # Get box weights
                    st.write("\n### Processing Box Weights")
                    try:
                        box_weights = get_box_weights(df, value_column, group_column, strain_column)
                        st.write("Box weights data:")
                        st.write(box_weights)
                        
                        if box_weights.empty:
                            st.error("No box weights calculated!")
                            st.stop()
                    except Exception as e:
                        st.error(f"Error calculating box weights: {str(e)}")
                        st.stop()
                    
                    # Run optimization
                    st.write("\n### Running Optimization")
                    try:
                        results = find_optimal_allocation_n_groups(box_weights, n_groups, group_names, strain_column)
                        st.write("Optimization results:")
                        st.write(results)
                        
                        if not results:
                            st.error("No results returned from optimization")
                            st.stop()
                    except Exception as e:
                        st.error(f"Optimization failed: {str(e)}")
                        st.write("Debug information:")
                        st.write("Box weights:")
                        st.write(box_weights)
                        st.stop()
                    
                    # Create output DataFrame
                    st.write("\n### Creating Output")
                    output_df = df.copy()
                    output_df['Allocated_Group'] = None
                    
                    # Process each strain
                    if strain_column:
                        strains = df[strain_column].unique()
                    else:
                        strains = ['Group']
                    
                    st.write(f"\nProcessing {len(strains)} strain(s): {strains}")
                    
                    # Track assignments for verification
                    assigned_boxes = set()
                    
                    for strain in strains:
                        st.write(f"\nProcessing strain: {strain}")
                        if strain not in results:
                            st.error(f"No results found for strain: {strain}")
                            continue
                        
                        strain_results = results[strain]
                        st.write(f"Results for strain {strain}:")
                        st.write(strain_results)
                        
                        for group_name, boxes in strain_results['groups'].items():
                            st.write(f"\nAssigning group {group_name}")
                            st.write(f"Boxes to assign: {boxes}")
                            
                            # Track these boxes
                            assigned_boxes.update(boxes)
                            
                            # Create mask using exact box numbers
                            mask = output_df[group_column].astype(str).isin([str(b) for b in boxes])
                            if strain_column:
                                mask &= (output_df[strain_column] == strain)
                            
                            # Debug: Show matching
                            st.write(f"Matching rows found: {mask.sum()}")
                            if mask.sum() == 0:
                                st.write("Box values in data:")
                                st.write(sorted(output_df[group_column].astype(str).unique()))
                                st.write("Looking for boxes:")
                                st.write(sorted([str(b) for b in boxes]))
                            
                            # Assign group
                            output_df.loc[mask, 'Allocated_Group'] = group_name
                    
                    # Verify assignments
                    st.write("\n### Verification")
                    st.write("Group assignments:")
                    st.write(output_df['Allocated_Group'].value_counts())
                    
                    # Check for unassigned boxes
                    all_boxes = set(output_df[group_column].astype(str).unique())
                    unassigned_boxes = all_boxes - assigned_boxes
                    if unassigned_boxes:
                        st.error(f"Found {len(unassigned_boxes)} unassigned boxes: {sorted(unassigned_boxes)}")
                        st.write("\nDebug Information")
                        st.write("1. All boxes in data:", sorted(all_boxes))
                        st.write("2. Assigned boxes:", sorted(assigned_boxes))
                        st.write("3. Box weights data:")
                        st.write(box_weights)
                        st.write("4. Results:")
                        st.write(results)
                        st.stop()
                    
                    # Display results
                    st.write("### Final Results")
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
                        df, 
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
