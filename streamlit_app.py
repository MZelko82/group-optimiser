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

def plot_group_distributions(df, results, value_col, strain_col=None):
    """Create distribution plots for the groups."""
    # Set up colors - alternating between first and last from each palette
    palette1 = ['#89A8B2', '#F1F0E8']  # First palette
    palette2 = ['#F0A8D0', '#FFEBD4']  # Second palette
    
    # Create a figure with transparent background
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8), facecolor='none')
    ax = fig.add_subplot(111)
    ax.set_facecolor('none')
    
    # Get group assignments
    df_plot = df.copy()
    
    # Process each strain
    if strain_col:
        strains = df[strain_col].unique()
    else:
        strains = ['Group']
    
    # Create plot data
    plot_data = []
    plot_groups = []
    
    for strain in strains:
        if strain not in results:
            continue
        for group, boxes in results[strain]['groups'].items():
            # Convert boxes to strings for comparison
            box_strings = [str(b) for b in boxes]
            mask = df_plot[value_col].index.isin(
                df_plot[df_plot['Box'].astype(str).isin(box_strings)].index
            )
            values = df_plot[value_col][mask]
            plot_data.extend(values)
            plot_groups.extend([group] * len(values))
    
    plot_df = pd.DataFrame({
        'Weight': plot_data,
        'Group': plot_groups
    })
    
    # Get unique groups and assign colors
    unique_groups = plot_df['Group'].unique()
    n_groups = len(unique_groups)
    
    # Create color list by alternating between first and last colors
    colors = []
    for i in range(n_groups):
        if i < len(palette1) // 2:
            colors.append(palette1[i * 2])
        else:
            idx = (i - len(palette1) // 2) * 2
            if idx < len(palette2):
                colors.append(palette2[idx])
            else:
                # If we run out of colors, start over
                colors.append(colors[i % len(colors)])
    
    # Create color map
    color_map = dict(zip(unique_groups, colors))
    
    # Plot scatter points
    for i, (group, group_data) in enumerate(plot_df.groupby('Group')):
        # Add jitter to y position
        y_jitter = np.random.normal(i, 0.1, size=len(group_data))
        ax.scatter(group_data['Weight'], y_jitter, 
                  alpha=0.5, color=color_map[group], 
                  label=group, s=50)
        
        # Add density plot
        density = gaussian_kde(group_data['Weight'])
        xs = np.linspace(group_data['Weight'].min(), group_data['Weight'].max(), 200)
        ys = density(xs)
        # Scale the density plot
        ys = ys / ys.max() * 0.5  # Scale to half the distance between groups
        ax.fill_between(xs, i - ys, i + ys, alpha=0.3, color=color_map[group])
    
    # Customize the plot
    ax.set_yticks(range(len(unique_groups)))
    ax.set_yticklabels(unique_groups, color='white', fontsize=10)
    
    # Style the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    ax.set_title('Weight Distribution by Group', color='white', pad=20, fontsize=12)
    ax.set_xlabel('Weight', color='white', fontsize=10)
    
    # Add grid with low opacity
    ax.grid(True, alpha=0.1, color='white')
    
    plt.tight_layout()
    return fig

def main():
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
                            
                            if not strain_results['groups']:
                                st.error(f"No group assignments found for strain {strain}")
                                continue
                            
                            for group_name, boxes in strain_results['groups'].items():
                                st.write(f"\nAssigning group {group_name}")
                                st.write(f"Boxes to assign: {boxes}")
                                
                                if not boxes:
                                    st.warning(f"No boxes to assign for group {group_name}")
                                    continue
                                
                                # Track these boxes
                                assigned_boxes.update(str(b) for b in boxes)
                                
                                # Create mask using exact box numbers
                                box_strings = [str(b) for b in boxes]
                                mask = output_df[group_column].astype(str).isin(box_strings)
                                if strain_column:
                                    mask &= (output_df[strain_column] == strain)
                                
                                # Debug: Show matching
                                st.write(f"Matching rows found: {mask.sum()}")
                                if mask.sum() == 0:
                                    st.write("\nBox number debug:")
                                    st.write("Box values in data:")
                                    data_boxes = sorted(output_df[group_column].astype(str).unique())
                                    st.write(data_boxes)
                                    st.write("Looking for boxes:")
                                    st.write(sorted(box_strings))
                                    st.write("\nBox types:")
                                    st.write(f"Data box type: {type(data_boxes[0])}")
                                    st.write(f"Assignment box type: {type(box_strings[0])}")
                                
                                # Assign group
                                output_df.loc[mask, 'Allocated_Group'] = group_name
                        
                        # Verify assignments
                        st.write("\n### Verification")
                        st.write("Group assignments:")
                        assignment_counts = output_df['Allocated_Group'].value_counts()
                        st.write(assignment_counts)
                        
                        # Check for unassigned boxes
                        all_boxes = set(output_df[group_column].astype(str).unique())
                        unassigned_boxes = all_boxes - assigned_boxes
                        if unassigned_boxes:
                            st.error(f"Found {len(unassigned_boxes)} unassigned boxes")
                            st.write("\nDebug Information")
                            st.write("1. All boxes:", sorted(all_boxes))
                            st.write("2. Assigned boxes:", sorted(assigned_boxes))
                            st.write("3. Unassigned boxes:", sorted(unassigned_boxes))
                            st.write("\n4. Box weights data:")
                            st.write(box_weights)
                            st.write("\n5. Results structure:")
                            st.write(results)
                            st.write("\n6. Data types:")
                            st.write(f"Box column type: {output_df[group_column].dtype}")
                            st.write(f"First few box values: {output_df[group_column].head()}")
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
                        st.write("### Weight Distributions")
                        fig = plot_group_distributions(
                            df=df, 
                            results=results, 
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
                            if strain not in results:
                                continue
                            st.write(f"\n**{strain}**")
                            st.write("Group totals:")
                            for group_name, total in results[strain]['group_weights'].items():
                                st.write(f"- {group_name}: {total:.1f}")
                            st.write(f"Variance between groups: {results[strain]['variance']:.2f}")
                            st.write(f"Maximum difference between groups: {results[strain]['max_difference']:.2f}")
                    
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.write("Please check your data and try again.")
                        raise e
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
