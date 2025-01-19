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

def plot_group_distributions(df, results, value_col, box_col, strain_col=None):
    """Create distribution plots for the groups, with separate views for overall and by strain."""
    try:
        # Create plot data
        plot_data = []
        for strain, strain_results in results.items():
            for group, boxes in strain_results['groups'].items():
                box_strings = [str(b) for b in boxes]
                if strain_col:
                    mask = (df[strain_col] == strain) & df[box_col].astype(str).isin(box_strings)
                else:
                    mask = df[box_col].astype(str).isin(box_strings)
                values = df.loc[mask, value_col]
                for value in values:
                    plot_data.append({
                        'Weight': value,
                        'Group': group,
                        'Strain': strain if strain_col else 'All'
                    })
        
        if not plot_data:
            st.error("No data available for plotting")
            return None
            
        plot_df = pd.DataFrame(plot_data)
        strains = plot_df['Strain'].unique()
        n_strains = len(strains)
        
        # Create figure
        with plt.style.context('dark_background'):
            fig, axes = plt.subplots(n_strains, 1, figsize=(12, 6 * n_strains), 
                                   facecolor='none', constrained_layout=True)
            if n_strains == 1:
                axes = [axes]
            
            # Set up colors
            unique_groups = plot_df['Group'].unique()
            n_groups = len(unique_groups)
            cmap = plt.cm.get_cmap('Set3')
            colors = [cmap(i/n_groups) for i in range(n_groups)]
            color_map = dict(zip(unique_groups, colors))
            
            # Create plots for each strain
            for i, strain in enumerate(strains):
                ax = axes[i]
                ax.set_facecolor('none')
                
                strain_data = plot_df[plot_df['Strain'] == strain]
                strain_groups = strain_data['Group'].unique()
                
                # Plot each group
                for j, group in enumerate(strain_groups):
                    group_data = strain_data[strain_data['Group'] == group]
                    weights = group_data['Weight']
                    
                    # Add jitter to y position
                    y_jitter = np.random.normal(j, 0.1, size=len(weights))
                    ax.scatter(weights, y_jitter, 
                              alpha=0.5, color=color_map[group], 
                              label=group, s=50)
                    
                    # Add density plot if we have enough points
                    if len(weights) > 1:
                        try:
                            density = gaussian_kde(weights)
                            xs = np.linspace(weights.min(), weights.max(), 200)
                            ys = density(xs)
                            ys = ys / ys.max() * 0.5
                            ax.fill_between(xs, j - ys, j + ys, alpha=0.3, color=color_map[group])
                        except Exception as e:
                            st.warning(f"Could not create density plot for {group} in {strain}: {str(e)}")
                
                # Customize plot
                ax.set_yticks(range(len(strain_groups)))
                ax.set_yticklabels(strain_groups, color='white', fontsize=10)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('white')
                ax.spines['bottom'].set_color('white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.set_title(f'Weight Distribution for {strain}', color='white', pad=20, fontsize=12)
                ax.set_xlabel('Weight', color='white', fontsize=10)
                ax.grid(True, alpha=0.1, color='white')
            
            return fig
            
    except Exception as e:
        st.error(f"Error creating plots: {str(e)}")
        import traceback
        st.write("Debug information:")
        st.write(traceback.format_exc())
        return None

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
                    
                    try:
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
                        # Add group weights
                        for group_name, total in result['group_weights'].items():
                            stats_data.append({
                                'Strain': strain,
                                'Group': group_name,
                                'Total Weight': f"{total:.1f}",
                                'Std Dev': f"{result['std_dev']:.2f}",  # Standard deviation within strain
                                'Max Difference': f"{result['max_difference']:.2f}"
                            })
                    
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df)
                    
                    # Create plots last with smaller size
                    st.write("### Weight Distributions")
                    fig = plot_group_distributions(df, st.session_state.results, value_column, group_column, strain_column)
                    if fig is not None:
                        # Make figure 20% smaller
                        fig.set_size_inches(fig.get_size_inches() * 0.8)
                        st.pyplot(fig)
                        
                    # Provide download link last
                    output = BytesIO()
                    st.session_state.output_df.to_csv(output, index=False)
                    output.seek(0)
                    st.download_button(
                        label="Download Results CSV",
                        data=output,
                        file_name="optimized_groups.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
