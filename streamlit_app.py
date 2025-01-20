import streamlit as st
import pandas as pd
import numpy as np
from optimizer import find_optimal_allocation_n_groups, get_box_weights
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from scipy.stats import gaussian_kde

# Configure seaborn defaults
sns.set_theme(style="whitegrid")

def plot_group_distributions(df, results, value_column, group_column, strain_column=None):
    """Plot weight distributions for each group, separated by strain."""
    if results is None:
        return None
    
    # Set style
    plt.style.use('dark_background')
    
    # Create separate plots for each strain
    strains = list(results.keys())
    if len(strains) == 0:
        return None
    
    # First create individual strain plots
    figs = []
    for strain in strains:
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get data for this strain
        strain_mask = df[strain_column] == strain if strain_column else pd.Series(True, index=df.index)
        strain_data = df[strain_mask]
        
        # Plot groups
        for group_name in results[strain]['groups'].keys():
            # Get boxes for this group
            box_strings = [str(b) for b in results[strain]['groups'][group_name]]
            group_mask = strain_data[group_column].astype(str).isin(box_strings)
            group_values = strain_data.loc[group_mask, value_column]
            
            # Create violin plot
            parts = ax.violinplot(group_values, positions=[list(results[strain]['groups'].keys()).index(group_name)],
                                showmeans=True, showmedians=True)
            
            # Customize violin plot colors
            for pc in parts['bodies']:
                pc.set_facecolor('lightblue')
                pc.set_alpha(0.7)
            parts['cmeans'].set_color('red')
            parts['cmedians'].set_color('white')
        
        # Customize plot
        ax.set_xticks(range(len(results[strain]['groups'])))
        ax.set_xticklabels(results[strain]['groups'].keys())
        ax.set_title(f'{strain} Weight Distribution by Group')
        ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3)
        
        figs.append(fig)
    
    # Now create a combined plot
    fig_combined, ax_combined = plt.subplots(figsize=(12, 6))
    
    # Calculate total number of groups and spacing
    total_groups = sum(len(results[strain]['groups']) for strain in strains)
    group_width = 0.8
    strain_spacing = 2  # Space between strains
    
    # Plot each strain's groups
    current_position = 0
    all_positions = []
    all_labels = []
    
    for strain_idx, strain in enumerate(strains):
        strain_data = df[df[strain_column] == strain] if strain_column else df
        
        # Plot each group in this strain
        for group_name in results[strain]['groups'].keys():
            # Get boxes for this group
            box_strings = [str(b) for b in results[strain]['groups'][group_name]]
            group_mask = strain_data[group_column].astype(str).isin(box_strings)
            group_values = strain_data.loc[group_mask, value_column]
            
            # Create violin plot
            parts = ax_combined.violinplot(group_values, positions=[current_position],
                                         showmeans=True, showmedians=True)
            
            # Customize violin plot colors
            for pc in parts['bodies']:
                pc.set_facecolor(['lightblue', 'lightgreen'][strain_idx % 2])
                pc.set_alpha(0.7)
            parts['cmeans'].set_color('red')
            parts['cmedians'].set_color('white')
            
            all_positions.append(current_position)
            all_labels.append(f"{strain}\n{group_name}")
            current_position += 1
        
        # Add spacing between strains if not the last strain
        if strain_idx < len(strains) - 1:
            current_position += strain_spacing
    
    # Customize combined plot
    ax_combined.set_xticks(all_positions)
    ax_combined.set_xticklabels(all_labels, rotation=45)
    ax_combined.set_title('Combined Weight Distribution by Strain and Group')
    ax_combined.set_ylabel('Weight')
    ax_combined.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return figs, fig_combined

def main():
    st.set_page_config(page_title="Group Optimizer", layout="wide")
    st.title("Group Allocation Optimizer")
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'output_df' not in st.session_state:
        st.session_state.output_df = None
    if 'optimization_run' not in st.session_state:
        st.session_state.optimization_run = False
        
    st.write("""
    Upload your data file (CSV or Excel) and optimize group allocations based on a numeric column while keeping specified groups together.
    The app uses Integer Linear Programming (ILP) to find the mathematically optimal solution that minimizes differences between groups.
    Note: Larger datasets with many boxes and groups may take longer to process.
    """)
    
    try:
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
                
                # Show column names for reference
                st.write("Available columns:", df.columns.tolist())
                
                with col1:
                    value_column = st.selectbox(
                        "Select the column to optimize groups on (numeric values):",
                        options=numeric_columns,
                        format_func=lambda x: f"{x} (numeric)"
                    )
                
                with col2:
                    group_column = st.selectbox(
                        "Select the column that identifies which items must stay together:",
                        options=df.columns.tolist(),
                        format_func=lambda x: f"{x} ({df[x].dtype})"
                    )
                
                strain_column = None
                if len(df.columns) > 2:  # If there are more columns, allow strain selection
                    strain_column = st.selectbox(
                        "Optional: Select a column to separate optimizations by (e.g., strain/type):",
                        options=['None'] + df.columns.tolist(),
                        format_func=lambda x: x if x == 'None' else f"{x} ({df[x].dtype})"
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
                
                if st.button("Run Optimization"):
                    # Clear previous results
                    st.session_state.results = None
                    st.session_state.output_df = None
                    st.session_state.optimization_run = False
                    
                    try:
                        # Running optimization section
                        with st.expander("Running Optimization", expanded=False):
                            st.write("The optimizer is running to find the best allocation that minimizes weight differences between groups.")
                            st.write("\nOptimizing for the following strains:")
                            if strain_column:
                                strains = df[strain_column].unique()
                            else:
                                strains = ['Group']
                            for strain in strains:
                                st.write(f"- {strain}")
                            
                            # Calculate box weights for all data
                            box_weights = get_box_weights(
                                df=df,
                                value_col=value_column,
                                box_col=group_column,
                                strain_col=strain_column
                            )
                            
                            # Debug output
                            st.write("Box weights:")
                            st.write(box_weights)
                            
                            # Single optimization call for all strains
                            st.session_state.results = find_optimal_allocation_n_groups(
                                box_weights=box_weights,
                                n_groups=n_groups,
                                group_names=group_names,
                                strain_col=strain_column
                            )
                            
                            # Debug output
                            st.write("Optimization results:")
                            st.write(st.session_state.results)
                            
                            # Create output DataFrame
                            st.session_state.output_df = df.copy()
                            st.session_state.output_df['Allocated_Group'] = None
                            
                            # Assign groups
                            all_boxes = set(df[group_column].astype(str))
                            assigned_boxes = set()
                            
                            # Debug output
                            st.write("Processing results for group assignment:")
                            st.write(st.session_state.results)
                            
                            for strain, result in st.session_state.results.items():
                                st.write(f"Processing strain {strain}:")
                                st.write(result)
                                for group, boxes in result['groups'].items():
                                    st.write(f"Assigning group {group} with boxes {boxes}")
                                    box_strings = [str(b) for b in boxes]
                                    strain_mask = df[strain_column] == strain if strain_column else pd.Series(True, index=df.index)
                                    mask = strain_mask & df[group_column].astype(str).isin(box_strings)
                                    st.session_state.output_df.loc[mask, 'Allocated_Group'] = group
                                    assigned_boxes.update(box_strings)
                            
                            # Check for unassigned boxes
                            unassigned = all_boxes - assigned_boxes
                            if unassigned:
                                st.warning(f"Warning: Some boxes were not assigned to any group: {unassigned}")
                            
                            st.session_state.optimization_run = True
                    
                    except Exception as e:
                        st.error(f"Error in optimization: {str(e)}")
                        raise e
                
                # Display results if optimization was successful
                if st.session_state.optimization_run:
                    # Show group summary
                    st.write("### Group Summary")
                    group_summary = st.session_state.output_df.groupby('Allocated_Group').agg({
                        value_column: ['count', 'sum', 'mean'],
                        group_column: lambda x: ', '.join(sorted(set(x.astype(str))))
                    }).reset_index()
                    group_summary.columns = ['Group', 'Subjects', 'Total Weight', 'Mean Weight', 'Boxes']
                    st.dataframe(group_summary)
                    
                    # Show full allocation
                    st.write("### Full Allocation")
                    st.dataframe(st.session_state.output_df)
                    
                    # Display statistics in a table
                    st.write("### Group Statistics")
                    stats_data = []
                    for strain, result in st.session_state.results.items():
                        # Calculate standard deviation per group after allocation
                        group_stdevs = {}
                        for group_name in result['group_weights'].keys():
                            group_mask = (st.session_state.output_df['Allocated_Group'] == group_name)
                            if strain_column:
                                group_mask &= (st.session_state.output_df[strain_column] == strain)
                            group_values = st.session_state.output_df.loc[group_mask, value_column]
                            group_stdevs[group_name] = float(np.std(group_values, ddof=1)) if len(group_values) > 1 else 0.0
                        
                        # Add group weights and calculated stats
                        for group_name, total in result['group_weights'].items():
                            stats_data.append({
                                'Strain': strain,
                                'Group': group_name,
                                'Total Weight': f"{total:.1f}",
                                'Std Dev': f"{group_stdevs[group_name]:.2f}",  # Per-group standard deviation
                                'Max Difference': f"{result['max_difference']:.2f}"
                            })
                    
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df)
                    
                    # Create plots last with smaller size
                    st.write("### Weight Distributions")
                    plot_results = plot_group_distributions(df, st.session_state.results, value_column, group_column, strain_column)
                    
                    if plot_results is not None:
                        strain_figs, combined_fig = plot_results
                        
                        # Show strain-specific plots
                        for fig in strain_figs:
                            st.pyplot(fig, use_container_width=True)
                        
                        # Show combined plot
                        st.write("### Combined Weight Distribution")
                        st.pyplot(combined_fig, use_container_width=True)
                    
                    # Create a separate container for the download button
                    download_container = st.container()
                    with download_container:
                        output = BytesIO()
                        st.session_state.output_df.to_csv(output, index=False)
                        output.seek(0)
                        st.download_button(
                            label="Download Results CSV",
                            data=output,
                            file_name="optimized_groups.csv",
                            mime="text/csv",
                            key="download_button"
                        )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                raise e
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
