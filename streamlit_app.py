import streamlit as st
import pandas as pd
import numpy as np
from optimizer import find_optimal_allocation_n_groups, get_box_weights
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from scipy.stats import gaussian_kde

# Set page config first
st.set_page_config(page_title="Group Optimizer", layout="wide")

# Configure seaborn defaults
sns.set_theme(style="whitegrid")

# Custom CSS for dataframe styling
st.markdown("""
<style>
    /* Hide the index column in dataframes */
    .dataframe-container [data-testid="stDataFrame"] div[data-testid="stHorizontalBlock"] {
        gap: 0 !important;
    }
    
    /* Remove extra padding and make table compact */
    .dataframe {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Style the table header */
    .dataframe thead th {
        background-color: transparent !important;
        color: white !important;
        text-align: left !important;
        padding: 8px !important;
    }
    
    /* Style the table cells */
    .dataframe tbody td {
        background-color: transparent !important;
        color: white !important;
        text-align: left !important;
        padding: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

def plot_group_distributions(df, results, value_column, group_column, strain_column=None):
    """Plot weight distributions for each group, separated by strain."""
    if results is None:
        return None
    
    # Set style and color palette
    plt.style.use('dark_background')
    colors = ['#89A8B2', '#F1F0E8',  # Primary palette alternating first/last
             '#F0A8D0', '#FFEBD4']   # Secondary palette alternating first/last
    
    # Create separate plots for each strain
    strains = list(results.keys())
    if len(strains) == 0:
        return None
    
    # First create individual strain plots
    figs = []
    for strain in strains:
        # Create figure with transparent background
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='none')
        ax.set_facecolor('none')
        
        # Handle the new weighted optimization structure
        if strain in ['Combined', 'Group']:
            # For weighted optimization, use all data (no strain filtering)
            strain_data = df
        else:
            # For old optimization, filter by strain
            strain_mask = df[strain_column] == strain if strain_column else pd.Series(True, index=df.index)
            strain_data = df[strain_mask]
        
        # Prepare data for plotting
        plot_data = []
        for group_name in results[strain]['groups'].keys():
            # Get boxes for this group
            box_strings = [str(b) for b in results[strain]['groups'][group_name]]
            group_mask = strain_data[group_column].astype(str).isin(box_strings)
            group_values = strain_data.loc[group_mask, value_column]
            
            # Add to plot data
            for val in group_values:
                plot_data.append({
                    'Group': group_name,
                    'Weight': val
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create horizontal violin plot with points
        for idx, group in enumerate(plot_df['Group'].unique()):
            group_data = plot_df[plot_df['Group'] == group]
            color = colors[idx % len(colors)]
            
            # Plot violin with transparency and no statistics
            sns.violinplot(data=group_data, x='Weight', y='Group', ax=ax,
                         inner=None, color=color, alpha=0.3, orient='h',
                         saturation=1.0)
            
            # Plot points with same color
            sns.stripplot(data=group_data, x='Weight', y='Group', ax=ax,
                         color=color, alpha=0.5, size=4, jitter=True, orient='h')
        
        # Customize plot
        plot_title = f'{strain} {value_column} Distribution by Group' if strain_column else f'{value_column} Distribution by Group'
        ax.set_title(plot_title, color='white', pad=10)
        ax.set_xlabel(value_column, color='white')
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        
        plt.tight_layout()
        figs.append(fig)
    
    # Only create combined plot if we have a strain column
    if strain_column is not None and len(strains) > 1:
        fig_combined, ax_combined = plt.subplots(figsize=(12, 5), facecolor='none')
        ax_combined.set_facecolor('none')
        
        # Prepare data for combined plot
        plot_data_combined = []
        all_positions = []
        all_labels = []
        
        # First, create the complete list of positions and labels
        for strain_idx, strain in enumerate(strains):
            for group_name in results[strain]['groups'].keys():
                all_labels.append(f"{strain}\n{group_name}")
                all_positions.append(len(all_positions))
        
        # Now plot in reverse order to match the labels
        current_position = len(all_positions) - 1
        for strain_idx, strain in enumerate(reversed(strains)):
            # Handle the new weighted optimization structure
            if strain in ['Combined', 'Group']:
                strain_data = df  # Use all data for weighted optimization
            else:
                strain_data = df[df[strain_column] == strain] if strain_column else df
            strain_color_offset = strain_idx * 2
            
            for group_idx, group_name in enumerate(reversed(list(results[strain]['groups'].keys()))):
                # Get boxes for this group
                box_strings = [str(b) for b in results[strain]['groups'][group_name]]
                group_mask = strain_data[group_column].astype(str).isin(box_strings)
                group_values = strain_data.loc[group_mask, value_column]
                
                color = colors[(strain_color_offset + group_idx) % len(colors)]
                
                # Create violin plot with transparency and no statistics
                if len(group_values) > 0:
                    violin_parts = ax_combined.violinplot(group_values, positions=[current_position],
                                                        vert=False, showmeans=False, showmedians=False,
                                                        showextrema=False)
                    
                    # Style violin plot
                    for pc in violin_parts['bodies']:
                        pc.set_facecolor(color)
                        pc.set_alpha(0.3)
                    
                    # Add strip plot with same color
                    ax_combined.scatter(group_values, 
                                     [current_position + (np.random.random(len(group_values)) - 0.5) * 0.1],
                                     alpha=0.5, color=color, s=20)
                
                current_position -= 1
        
        # Customize combined plot
        ax_combined.set_yticks(range(len(all_labels)))
        ax_combined.set_yticklabels(all_labels, color='white')
        plot_title = f'Combined {value_column} Distribution by {strain_column} and Group'
        ax_combined.set_title(plot_title, color='white', pad=10)
        ax_combined.set_xlabel(value_column, color='white')
        ax_combined.grid(True, alpha=0.3)
        ax_combined.tick_params(colors='white')
        for spine in ax_combined.spines.values():
            spine.set_color('white')
        
        plt.tight_layout()
    else:
        fig_combined = None
    
    return figs, fig_combined

def plot_initial_distribution(df, value_column, strain_column=None):
    """Plot initial weight distribution for the full dataset and by strain."""
    plt.style.use('dark_background')
    colors = ['#89A8B2', '#F1F0E8',  # Primary palette alternating first/last
             '#F0A8D0', '#FFEBD4']   # Secondary palette alternating first/last
    
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='none')
    ax.set_facecolor('none')
    
    # Prepare data for plotting
    plot_data = []
    
    # Add full dataset
    plot_data.append({
        'label': 'Full Dataset',
        'values': df[value_column],
        'position': 0,
        'color': colors[0]
    })
    
    # Add strain-specific data if strain column is provided
    if strain_column is not None:
        for i, strain in enumerate(sorted(df[strain_column].unique())):
            strain_data = df[df[strain_column] == strain][value_column]
            plot_data.append({
                'label': strain,
                'values': strain_data,
                'position': i + 1,
                'color': colors[(i + 1) % len(colors)]
            })
    
    # Create violin plots with points
    positions = []
    labels = []
    for data in plot_data:
        pos = data['position']
        positions.append(pos)
        labels.append(data['label'])
        
        # Create violin plot
        violin_parts = ax.violinplot(data['values'], positions=[pos],
                                   vert=False, showmeans=False, showmedians=False,
                                   showextrema=False)
        
        # Style violin plot
        for pc in violin_parts['bodies']:
            pc.set_facecolor(data['color'])
            pc.set_alpha(0.3)
        
        # Add strip plot
        ax.scatter(data['values'],
                  [pos + (np.random.random(len(data['values'])) - 0.5) * 0.1],
                  alpha=0.5, color=data['color'], s=20)
    
    # Customize plot
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, color='white')
    ax.set_title('Initial Weight Distribution', color='white')
    ax.set_xlabel('Weight', color='white')
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.tight_layout()
    return fig

def create_birthdate_plot(df, value_column, strain_column):
    """Create a plot showing birthdate distribution within each group."""
    plt.style.use('dark_background')
    colors = ['#89A8B2', '#F1F0E8', '#F0A8D0', '#FFEBD4']
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='none')
    ax.set_facecolor('none')
    
    # Get unique birthdates and groups
    birthdates = sorted(df[strain_column].unique())
    groups = sorted(df['Allocated_Group'].unique())
    
    # Create position mapping
    x_positions = []
    labels = []
    colors_list = []
    
    position = 0
    for group in groups:
        for birthdate in birthdates:
            # Get data for this group and birthdate combination
            group_birthdate_data = df[(df['Allocated_Group'] == group) & (df[strain_column] == birthdate)]
            
            if len(group_birthdate_data) > 0:
                values = group_birthdate_data[value_column]
                
                # Create violin plot
                violin_parts = ax.violinplot(values, positions=[position],
                                           vert=False, showmeans=False, showmedians=False,
                                           showextrema=False)
                
                # Style violin plot
                color_idx = groups.index(group) % len(colors)
                for pc in violin_parts['bodies']:
                    pc.set_facecolor(colors[color_idx])
                    pc.set_alpha(0.3)
                
                # Add scatter plot
                ax.scatter(values, [position + (np.random.random(len(values)) - 0.5) * 0.1],
                          alpha=0.6, color=colors[color_idx], s=20)
                
                x_positions.append(position)
                labels.append(f"{group}\n{birthdate}")
                colors_list.append(colors[color_idx])
                
                position += 1
    
    # Customize plot
    ax.set_yticks(x_positions)
    ax.set_yticklabels(labels, color='white')
    ax.set_title(f'{value_column} Distribution by Group and {strain_column}', color='white', pad=10)
    ax.set_xlabel(value_column, color='white')
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    # Add legend
    group_handles = []
    for i, group in enumerate(groups):
        group_handles.append(plt.Rectangle((0,0),1,1, facecolor=colors[i % len(colors)], alpha=0.7))
    ax.legend(group_handles, groups, loc='upper right', title='Groups')
    
    plt.tight_layout()
    return fig

def main():
    st.title("Group Allocation Optimizer")
    
    # Initialize session state variables
    if 'optimization_run' not in st.session_state:
        st.session_state.optimization_run = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'output_df' not in st.session_state:
        st.session_state.output_df = None
    
    # Show example data preview
    st.markdown("### Example Data Format")
    try:
        # Read and prepare example data
        example_df = pd.read_csv('example_data.csv')
        # Select columns in desired order
        display_df = example_df[['ID', 'Box', 'Strain', 'Weight']].copy()
        
        # Create a styled dataframe
        styled_df = display_df.style.set_properties(**{
            'background-color': 'transparent',
            'color': 'white'
        })
        
        # Display the dataframe with custom styling
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.dataframe(
            styled_df,
            hide_index=True,
            use_container_width=False
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add link to strain example CSV
        st.markdown('[Download strain example csv](https://github.com/MZelko82/group-optimiser/blob/main/example_data2.csv)')
    except Exception as e:
        st.warning("Example data file not found. Please ensure example_data.csv is in the same directory.")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("File uploaded successfully!")
            
            # Get available columns for selection
            columns = list(df.columns)
            
            # Create three columns for the dropdowns
            st.markdown("### Select Columns for Optimization")
            
            # Column to optimize
            st.markdown("##### Which column should be optimized for even distribution?")
            value_column = st.selectbox("", df.select_dtypes(include=[np.number]).columns.tolist(), key='optimize')
            
            # Column for keeping subjects together
            st.markdown("##### Which column indicates which animals should remain in the same group (Box, cage etc)?")
            group_column = st.selectbox("", df.columns.tolist(), key='stay_together')
            
            # Check if the selected group column has unique values for each row
            if df[group_column].nunique() == len(df):
                st.warning("⚠️ The selected grouping column has a unique value for each row. This means each animal will be treated as its own group. If you want to keep multiple animals together, please select a column where some values are shared between rows (like Box or Cage numbers).")
            
            # Optional strain column
            st.markdown("##### Which column indicates the grouping variable (Strain, Age etc) prior to experimental group allocation?")
            strain_column = st.selectbox("", ['None'] + df.columns.tolist(), key='strain')
            strain_column = None if strain_column == 'None' else strain_column
            
            # Display available columns for reference
            st.markdown("#### Available Columns:")
            st.write(", ".join(columns))
            
            # Group configuration
            st.write("### Group Configuration")
            n_groups = st.number_input("Number of groups", min_value=2, max_value=10, value=2)
            
            # Group names input
            st.write("#### Group Labels")
            group_names = []
            cols = st.columns(min(n_groups, 4))  # Show up to 4 columns
            for i in range(n_groups):
                col_idx = i % 4
                with cols[col_idx]:
                    group_name = st.text_input(f"Group {i+1} Label:", value=f"Group {i+1}")
                    group_names.append(group_name)
            
            # Check for duplicate group names
            if len(set(group_names)) != len(group_names):
                st.error("Please ensure all group names are unique!")
                st.stop()
            
            # Run optimization
            if st.button("Optimize Groups"):
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
                            group_names=group_names,  # Use the user-defined group names
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
                                
                                # Handle the new weighted optimization structure
                                if strain in ['Combined', 'Group']:
                                    # For weighted optimization, don't filter by strain
                                    strain_mask = pd.Series(True, index=df.index)
                                else:
                                    # For old optimization, filter by strain
                                    strain_mask = df[strain_column] == strain if strain_column else pd.Series(True, index=df.index)
                                
                                mask = strain_mask & df[group_column].astype(str).isin(box_strings)
                                st.session_state.output_df.loc[mask, 'Allocated_Group'] = group
                                assigned_boxes.update(box_strings)
                                
                                # Debug: Check how many animals were assigned
                                assigned_count = mask.sum()
                                st.write(f"  → Assigned {assigned_count} animals to {group}")
                        
                        # Check for unassigned boxes
                        unassigned = all_boxes - assigned_boxes
                        if unassigned:
                            st.warning(f"Warning: Some boxes were not assigned to any group: {unassigned}")
                        
                        # Debug: Check final state of Allocated_Group column
                        assigned_animals = st.session_state.output_df['Allocated_Group'].notna().sum()
                        total_animals = len(st.session_state.output_df)
                        st.write(f"Final assignment: {assigned_animals}/{total_animals} animals assigned to groups")
                        
                        # Show sample of assignments
                        sample_assignments = st.session_state.output_df[['Rat ID', 'Rat Box', 'Allocated_Group']].head(10)
                        st.write("Sample assignments:")
                        st.write(sample_assignments)
                        
                        st.session_state.optimization_run = True
                
                except Exception as e:
                    st.error(f"Error during optimization: {str(e)}")
            
            # Display results if optimization was successful
            if st.session_state.optimization_run:
                # Show initial distribution
                st.write("### Initial Distribution")
                initial_fig = plot_initial_distribution(df, value_column, strain_column)
                st.pyplot(initial_fig)
                
                # Check if we're using the new weighted optimization
                is_weighted_optimization = strain_column and 'Combined' in st.session_state.results.keys()
                
                # 1. Group Summary and Plot
                st.write("### Group Summary")
                group_summary = st.session_state.output_df.groupby('Allocated_Group').agg({
                    value_column: ['count', 'mean', 'std']
                }).round(2)
                st.write(group_summary)
                
                st.write("### Group Distribution Plot")
                group_plot_results = plot_group_distributions(df, st.session_state.results, value_column, group_column, strain_column)
                
                if group_plot_results is not None:
                    strain_figs, combined_fig = group_plot_results
                    if strain_figs:
                        for fig in strain_figs:
                            st.pyplot(fig)
                        if combined_fig is not None:
                            st.pyplot(combined_fig)
                
                # 2. Birthdate Summary and Plot (if strain column exists)
                if strain_column:
                    st.write("### Group Summary by " + strain_column)
                    birthdate_summary = st.session_state.output_df.groupby([strain_column, 'Allocated_Group']).agg({
                        value_column: ['count', 'mean', 'std']
                    }).round(2)
                    st.write(birthdate_summary)
                    
                    st.write("### " + strain_column + " Distribution by Group")
                    # Create a birthdate-specific plot
                    birthdate_plot_fig = create_birthdate_plot(st.session_state.output_df, value_column, strain_column)
                    if birthdate_plot_fig:
                        st.pyplot(birthdate_plot_fig)
                
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
            st.error(f"Error reading file: {str(e)}")

if __name__ == "__main__":
    main()
