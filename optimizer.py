import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import itertools

def get_box_weights(df, value_col, box_col, strain_col=None):
    """Calculate total value for each box."""
    if strain_col:
        return df.groupby([strain_col, box_col])[value_col].sum().reset_index()
    return df.groupby([box_col])[value_col].sum().reset_index()

def find_optimal_allocation(box_weights, strain_col=None):
    """Find the optimal allocation of boxes to minimize value difference between groups."""
    results = {}
    
    if strain_col:
        strains = box_weights[strain_col].unique()
    else:
        strains = ['Group']
        strain_col = 'dummy'
        box_weights['dummy'] = 'Group'
    
    for strain in strains:
        strain_boxes = box_weights[box_weights[strain_col] == strain]
        boxes = strain_boxes[strain_boxes.columns[1]].tolist()  # box column is always second
        n_boxes = len(boxes)
        n_boxes_per_group = n_boxes // 2
        
        min_diff = float('inf')
        optimal_allocation = None
        
        # Generate all possible combinations for the first group
        for combo in itertools.combinations(boxes, n_boxes_per_group):
            combo_value = strain_boxes[strain_boxes[strain_boxes.columns[1]].isin(combo)][strain_boxes.columns[2]].sum()
            complement = [box for box in boxes if box not in combo]
            complement_value = strain_boxes[strain_boxes[strain_boxes.columns[1]].isin(complement)][strain_boxes.columns[2]].sum()
            
            diff = abs(combo_value - complement_value)
            
            if diff < min_diff:
                min_diff = diff
                optimal_allocation = {
                    'CON': sorted(combo),
                    'CR': sorted(complement),
                    'CON_weight': combo_value,
                    'CR_weight': complement_value,
                    'weight_difference': diff
                }
        
        results[strain] = optimal_allocation
    
    return results

def plot_group_distributions(df, results, value_col='Weight', strain_col=None):
    """Create combined density and scatter plots for each strain and group."""
    # Create a new column for group assignments
    df['Group'] = 'Unassigned'
    
    if strain_col is None:
        strain_col = 'dummy'
        df['dummy'] = 'Group'
        strains = ['Group']
    else:
        strains = df[strain_col].unique()
    
    # Set consistent colors for groups
    colors = {'CON': '#2ecc71', 'CR': '#e74c3c'}
    
    # Create the plots
    n_strains = len(strains)
    fig, axes = plt.subplots(1, n_strains, figsize=(7*n_strains, 5))
    if n_strains == 1:
        axes = [axes]
    
    # Create separate plots for each strain
    for i, strain in enumerate(strains):
        strain_data = df[df[strain_col] == strain]
        
        # Assign groups based on results
        for box in results[strain]['CON']:
            mask = strain_data['Rat Box'] == box
            df.loc[df.index[mask], 'Group'] = 'CON'
        for box in results[strain]['CR']:
            mask = strain_data['Rat Box'] == box
            df.loc[df.index[mask], 'Group'] = 'CR'
        
        strain_data = df[df[strain_col] == strain]
        
        # Density plot
        sns.kdeplot(data=strain_data, 
                   x=value_col, 
                   hue='Group',
                   fill=True,
                   alpha=0.5,
                   palette=colors,
                   ax=axes[i])
        
        # Calculate and display group means
        group_means = strain_data.groupby('Group')[value_col].mean()
        ymax = axes[i].get_ylim()[1]
        axes[i].set_ylim(0, ymax * 1.2)
        
        # Calculate x-axis range for offset
        xmin, xmax = axes[i].get_xlim()
        x_range = xmax - xmin
        offset = x_range * 0.03
        
        for group in ['CON', 'CR']:
            axes[i].axvline(group_means[group], 
                          color=colors[group],
                          linestyle='--',
                          alpha=0.8)
            label_x = group_means[group] + (offset if group == 'CR' else -offset)
            axes[i].text(label_x, 
                        ymax * 1.1,
                        f'{group}\n{group_means[group]:.1f}',
                        horizontalalignment='center',
                        verticalalignment='center',
                        color=colors[group])
        
        # Add scatter plot at the bottom
        for group in ['CON', 'CR']:
            group_data = strain_data[strain_data['Group'] == group]
            axes[i].scatter(group_data[value_col], 
                          [-0.02] * len(group_data),
                          c=[colors[group]],
                          alpha=0.6,
                          s=100,
                          label=group)
        
        axes[i].legend()
        axes[i].set_title(f'{strain}')
        axes[i].set_xlabel(value_col)
        if i == 0:
            axes[i].set_ylabel('Density')
        else:
            axes[i].set_ylabel('')
    
    plt.tight_layout()
    return fig
