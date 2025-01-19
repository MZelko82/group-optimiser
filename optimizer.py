import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import itertools
from typing import List, Dict, Any
from pulp import *

def get_box_weights(df, value_col, box_col, strain_col=None):
    """Calculate total value for each box."""
    # Ensure value column is numeric
    df = df.copy()
    print(f"Input data types: {df.dtypes}")
    print(f"Sample data:\n{df.head()}")
    
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    print(f"After conversion:\n{df.head()}")
    
    # Get the number of subjects per box for allocation
    box_counts = df.groupby(box_col).size().reset_index(name='subjects_per_box')
    print(f"Subjects per box:\n{box_counts}")
    
    # Calculate total weight per box (we'll use this for balancing)
    if strain_col:
        box_totals = df.groupby([strain_col, box_col])[value_col].sum().reset_index()
        # Add the count information
        box_totals = box_totals.merge(box_counts, on=box_col)
    else:
        box_totals = df.groupby(box_col)[value_col].sum().reset_index()
        # Add the count information
        box_totals = box_totals.merge(box_counts, on=box_col)
    
    print(f"Box totals with subject counts:\n{box_totals}")
    return box_totals

def find_optimal_allocation_ilp(boxes: List[str], values: Dict[str, float], n_groups: int, group_names: List[str], subjects_per_box: Dict[str, int]) -> Dict:
    """
    Find the optimal allocation of boxes to groups using Integer Linear Programming.
    This ensures the most balanced distribution possible by minimizing the maximum difference between groups.
    """
    print(f"\nStarting ILP optimization")
    print(f"Number of boxes: {len(boxes)}")
    print(f"Number of groups: {n_groups}")
    print(f"Group names: {group_names}")
    print(f"Subjects per box: {subjects_per_box}")
    
    # Convert values to float, ensuring we only convert the numeric values
    values = {str(k): float(v) for k, v in values.items()}
    boxes = [str(box) for box in boxes]  # Ensure all boxes are strings
    
    print(f"Processed values: {values}")
    
    # Create the model
    prob = LpProblem("GroupAllocation", LpMinimize)
    
    # Decision variables: x[i,j] = 1 if box i is assigned to group j
    x = LpVariable.dicts("assign",
                        ((box, group) for box in boxes for group in group_names),
                        cat='Binary')
    
    # Variable for the maximum group total (to minimize)
    max_group_total = LpVariable("max_group_total")
    # Variable for the minimum group total (to minimize difference)
    min_group_total = LpVariable("min_group_total")
    
    # Objective: Minimize the difference between max and min group totals
    prob += max_group_total - min_group_total
    
    # Constraints
    # 1. Each box must be assigned to exactly one group
    for box in boxes:
        prob += lpSum(x[box, group] for group in group_names) == 1
    
    # 2. Groups should have approximately equal number of subjects (not boxes)
    total_subjects = sum(subjects_per_box.values())
    subjects_per_group = total_subjects // n_groups
    remainder = total_subjects % n_groups
    
    print(f"Total subjects: {total_subjects}")
    print(f"Subjects per group: {subjects_per_group}")
    print(f"Remainder: {remainder}")
    
    # Calculate min and max subjects per group
    min_subjects = subjects_per_group
    max_subjects = subjects_per_group + (1 if remainder > 0 else 0)
    
    print(f"Min subjects per group: {min_subjects}")
    print(f"Max subjects per group: {max_subjects}")
    
    # Allow some flexibility in group sizes to ensure feasibility
    for group in group_names:
        group_subjects = lpSum(subjects_per_box[box] * x[box, group] for box in boxes)
        # Groups must be within ±1 of target size
        prob += group_subjects >= min_subjects
        prob += group_subjects <= max_subjects
    
    # 3. Track max and min group totals (for weights)
    group_totals = {}
    for group in group_names:
        group_total = lpSum(values[box] * x[box, group] for box in boxes)
        group_totals[group] = group_total
        prob += group_total <= max_group_total
        prob += group_total >= min_group_total
    
    # Solve the problem
    print("\nSolving optimization problem...")
    status = prob.solve(PULP_CBC_CMD(msg=False))
    print(f"Solution status: {LpStatus[prob.status]}")
    
    if LpStatus[prob.status] != 'Optimal':
        # Get more detailed debug information
        group_constraints = {}
        for group in group_names:
            group_subjects = sum(subjects_per_box[box] for box in boxes if value(x[box, group]) > 0.5)
            group_constraints[group] = {
                'subjects': group_subjects,
                'min_required': min_subjects,
                'max_allowed': max_subjects,
                'boxes': [box for box in boxes if value(x[box, group]) > 0.5]
            }
        
        raise ValueError(f"Could not find optimal solution. Status: {LpStatus[prob.status]}\n"
                        f"Debug info:\n"
                        f"Group constraints: {group_constraints}\n"
                        f"Total subjects: {total_subjects}\n"
                        f"Target per group: {subjects_per_group} (±{1 if remainder > 0 else 0})")
    
    # Extract results
    allocation = {group: [] for group in group_names}
    final_totals = {group: 0 for group in group_names}
    assignments = {}  # Track assignments for debugging
    
    # First pass: get all definite assignments
    for box in boxes:
        # Get assignment values for each group
        group_values = {group: value(x[box, group]) for group in group_names}
        assignments[box] = group_values
        print(f"Box {box} assignments: {group_values}")
        
        # Find the group with the highest assignment value
        best_group = max(group_values.items(), key=lambda x: x[1])[0]
        allocation[best_group].append(box)
        final_totals[best_group] += values[box]
    
    print("\nFinal allocation:")
    for group, boxes in allocation.items():
        group_subjects = sum(subjects_per_box[box] for box in boxes)
        print(f"{group}: {boxes} ({group_subjects} subjects, Total weight: {final_totals[group]})")
    
    # Verify all boxes are assigned
    all_assigned_boxes = set().union(*[set(boxes) for boxes in allocation.values()])
    missing_boxes = set(boxes) - all_assigned_boxes
    if missing_boxes:
        # Debug information
        debug_info = {
            'missing_boxes': list(missing_boxes),
            'assignments': {box: {g: v for g, v in assignments[box].items() if v > 0.01} 
                          for box in missing_boxes},
            'group_sizes': {g: sum(subjects_per_box[box] for box in boxes) 
                          for g, boxes in allocation.items()},
            'target_sizes': {g: subjects_per_group + (1 if i < remainder else 0) 
                           for i, g in enumerate(group_names)}
        }
        raise ValueError(f"Not all boxes were assigned to groups. Debug info: {debug_info}")
    
    # Calculate statistics
    totals = list(final_totals.values())
    return {
        'groups': allocation,
        'group_weights': final_totals,
        'variance': float(np.var(totals)),
        'max_difference': float(max(totals) - min(totals))
    }

def find_optimal_allocation_n_groups(box_weights, n_groups: int, group_names: List[str], strain_col=None):
    """Find the optimal allocation of boxes to minimize value difference between N groups using ILP."""
    print(f"\nStarting allocation with {n_groups} groups: {group_names}")
    print(f"Box weights:\n{box_weights}")
    
    results = {}
    
    if strain_col:
        strains = box_weights[strain_col].unique()
    else:
        strains = ['Group']
        strain_col = 'dummy'
        box_weights['dummy'] = 'Group'
    
    print(f"Processing strains: {strains}")
    
    for strain in strains:
        print(f"\nProcessing strain: {strain}")
        strain_boxes = box_weights[box_weights[strain_col] == strain]
        print(f"Strain boxes:\n{strain_boxes}")
        
        # Get the column names for box and value columns
        box_col = strain_boxes.columns[1]  # box column is always second
        value_col = strain_boxes.columns[2]  # value column is always third
        
        print(f"Box column: {box_col}, Value column: {value_col}")
        
        # Convert boxes to strings and ensure values are numeric
        boxes = strain_boxes[box_col].astype(str).tolist()
        values = pd.to_numeric(strain_boxes[value_col], errors='coerce').fillna(0).tolist()
        box_values = dict(zip(boxes, values))
        subjects_per_box = dict(zip(boxes, strain_boxes['subjects_per_box'].tolist()))
        
        print(f"Boxes: {boxes}")
        print(f"Values: {values}")
        print(f"Box-value mapping: {box_values}")
        print(f"Subjects per box: {subjects_per_box}")
        
        try:
            results[strain] = find_optimal_allocation_ilp(boxes, box_values, n_groups, group_names, subjects_per_box)
            print(f"Allocation successful for strain {strain}")
            print(f"Results: {results[strain]}")
        except Exception as e:
            print(f"Error in allocation for strain {strain}: {str(e)}")
            raise ValueError(f"Error optimizing allocation for {strain}: {str(e)}")
    
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
    
    # Generate colors for groups
    unique_groups = set()
    for strain_result in results.values():
        unique_groups.update(strain_result['groups'].keys())
    n_groups = len(unique_groups)
    colors = sns.color_palette("husl", n_groups)
    color_dict = dict(zip(sorted(unique_groups), colors))
    
    # Create the plots
    n_strains = len(strains)
    fig, axes = plt.subplots(1, n_strains, figsize=(7*n_strains, 5))
    if n_strains == 1:
        axes = [axes]
    
    # Create separate plots for each strain
    for i, strain in enumerate(strains):
        strain_data = df[df[strain_col] == strain]
        
        # Assign groups based on results
        for group_name, boxes in results[strain]['groups'].items():
            mask = strain_data['Rat Box'].isin(boxes)
            df.loc[df.index[mask], 'Group'] = group_name
        
        strain_data = df[df[strain_col] == strain]
        
        # Density plot
        sns.kdeplot(data=strain_data, 
                   x=value_col, 
                   hue='Group',
                   fill=True,
                   alpha=0.5,
                   palette=color_dict,
                   ax=axes[i])
        
        # Calculate and display group means
        group_means = strain_data.groupby('Group')[value_col].mean()
        ymax = axes[i].get_ylim()[1]
        axes[i].set_ylim(0, ymax * 1.2)
        
        # Calculate x-axis range for offset
        xmin, xmax = axes[i].get_xlim()
        x_range = xmax - xmin
        offset = x_range * 0.03
        
        # Add mean lines and labels for each group
        for j, (group, mean) in enumerate(group_means.items()):
            axes[i].axvline(mean, 
                          color=color_dict[group],
                          linestyle='--',
                          alpha=0.8)
            # Stagger labels vertically
            y_pos = ymax * (1.1 - j * 0.1)
            axes[i].text(mean + offset,
                        y_pos,
                        f'{group}\n{mean:.1f}',
                        horizontalalignment='left',
                        verticalalignment='center',
                        color=color_dict[group])
        
        # Add scatter plot at the bottom
        for group in group_means.index:
            group_data = strain_data[strain_data['Group'] == group]
            axes[i].scatter(group_data[value_col], 
                          [-0.02] * len(group_data),
                          c=[color_dict[group]],
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
