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
    box_totals = df.groupby(box_col)[value_col].sum().reset_index()
    
    # Merge counts and totals
    box_data = box_counts.merge(box_totals, on=box_col)
    
    print(f"Box data:\n{box_data}")
    return box_data

def find_optimal_allocation_ilp(boxes: List[str], values: Dict[str, float], n_groups: int, group_names: List[str], subjects_per_box: Dict[str, int]) -> Dict:
    """
    Find the optimal allocation of boxes to groups using Integer Linear Programming.
    Uses a two-step approach:
    1. First find a feasible solution that balances subject counts
    2. Then optimize weights while maintaining subject balance
    """
    print("\n=== Starting ILP Optimization ===")
    print(f"Input data:")
    print(f"- Boxes: {boxes}")
    print(f"- Values: {values}")
    print(f"- Subjects per box: {subjects_per_box}")
    print(f"- Number of groups: {n_groups}")
    print(f"- Group names: {group_names}")
    
    # Validate input data
    if not boxes or not values or not subjects_per_box:
        raise ValueError("Empty input data")
    if len(boxes) != len(values) or len(boxes) != len(subjects_per_box):
        raise ValueError(f"Mismatched input lengths: boxes={len(boxes)}, values={len(values)}, subjects={len(subjects_per_box)}")
    
    # Convert values to float, ensuring we only convert the numeric values
    values = {str(k): float(v) for k, v in values.items()}
    boxes = [str(box) for box in boxes]  # Ensure all boxes are strings
    subjects_per_box = {str(k): int(v) for k, v in subjects_per_box.items()}
    
    # Step 1: Find a feasible solution balancing subjects
    prob = LpProblem("GroupAllocation_Step1", LpMinimize)
    
    print("\nCreating decision variables...")
    # Decision variables: x[i,j] = 1 if box i is assigned to group j
    x = LpVariable.dicts("assign",
                        ((box, group) for box in boxes for group in group_names),
                        cat='Binary')
    
    # Variables for tracking subject count differences
    max_subjects = LpVariable("max_subjects", lowBound=0)
    min_subjects = LpVariable("min_subjects", lowBound=0)
    
    print("\nSetting up objective and constraints...")
    # Objective: Minimize the difference in subject counts
    prob += max_subjects - min_subjects
    
    # Constraints
    # 1. Each box must be assigned to exactly one group
    for box in boxes:
        prob += lpSum(x[box, group] for group in group_names) == 1
        # Debug: print constraint
        print(f"Box {box} assignment sum: {lpSum(x[box, group] for group in group_names)} = 1")
    
    # 2. Track min and max subjects per group
    total_subjects = sum(subjects_per_box.values())
    subjects_per_group = total_subjects // n_groups
    remainder = total_subjects % n_groups
    
    print(f"\nSubject distribution:")
    print(f"- Total subjects: {total_subjects}")
    print(f"- Target per group: {subjects_per_group}")
    print(f"- Remainder: {remainder}")
    
    # Allow more flexibility in group sizes
    min_allowed = subjects_per_group - 2
    max_allowed = subjects_per_group + 2
    
    print(f"Allowed group size range: {min_allowed} to {max_allowed}")
    
    # Track subjects per group
    group_subject_counts = {}
    for group in group_names:
        group_subjects = lpSum(subjects_per_box[box] * x[box, group] for box in boxes)
        group_subject_counts[group] = group_subjects
        # Track min/max
        prob += group_subjects <= max_subjects
        prob += group_subjects >= min_subjects
        # Keep within allowed range
        prob += group_subjects >= min_allowed
        prob += group_subjects <= max_allowed
        # Debug: print constraints
        print(f"Group {group} subjects: {min_allowed} <= {group_subjects} <= {max_allowed}")
    
    # Solve step 1
    print("\nSolving step 1: Subject balance...")
    status = prob.solve(PULP_CBC_CMD(msg=False))
    print(f"Step 1 status: {LpStatus[prob.status]}")
    
    if LpStatus[prob.status] != 'Optimal':
        print("\nStep 1 failed to find optimal solution")
        # Get debug information
        group_info = {}
        for group in group_names:
            assigned_boxes = [box for box in boxes if value(x[box, group]) > 0.5]
            group_info[group] = {
                'boxes': assigned_boxes,
                'subjects': sum(subjects_per_box[box] for box in assigned_boxes),
                'weight': sum(values[box] for box in assigned_boxes),
                'box_sizes': {box: subjects_per_box[box] for box in assigned_boxes}
            }
        
        print("\nDebug information:")
        print(f"Group info: {group_info}")
        raise ValueError(f"Could not find feasible subject balance. Status: {LpStatus[prob.status]}")
    
    # Extract results directly from step 1
    print("\nExtracting results from step 1...")
    allocation = {group: [] for group in group_names}
    final_totals = {group: 0 for group in group_names}
    final_subjects = {group: 0 for group in group_names}
    
    # Track all assigned boxes
    all_assigned_boxes = set()
    
    for box in boxes:
        # Find which group this box is assigned to
        assigned = False
        for group in group_names:
            if value(x[box, group]) > 0.5:  # Box is assigned to this group
                allocation[group].append(box)
                final_totals[group] += values[box]
                final_subjects[group] += subjects_per_box[box]
                all_assigned_boxes.add(box)
                assigned = True
                print(f"Assigned box {box} to group {group}")
                break
        
        if not assigned:
            print(f"Warning: Box {box} was not assigned to any group!")
    
    print("\nFinal allocation:")
    for group, boxes in allocation.items():
        print(f"{group}: {boxes} ({final_subjects[group]} subjects, Total weight: {final_totals[group]})")
    
    # Verify all boxes are assigned
    missing_boxes = set(boxes) - all_assigned_boxes
    if missing_boxes:
        print("\nMissing box assignments:")
        print(f"Total boxes: {len(boxes)}")
        print(f"Assigned boxes: {len(all_assigned_boxes)}")
        print(f"Missing boxes: {missing_boxes}")
        raise ValueError(f"Not all boxes were assigned! Missing: {missing_boxes}")
    
    # Calculate statistics
    totals = list(final_totals.values())
    results = {
        'groups': allocation,
        'group_weights': final_totals,
        'group_subjects': final_subjects,
        'variance': float(np.var(totals)),
        'max_difference': float(max(totals) - min(totals))
    }
    
    print("\nOptimization complete!")
    print(f"Results: {results}")
    return results

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
        box_col = strain_boxes.columns[0]  # First column is box
        value_col = strain_boxes.columns[2]  # Third column is total weight
        
        print(f"Box column: {box_col}, Value column: {value_col}")
        
        # Convert boxes and values to dictionaries, preserving actual box numbers
        boxes = strain_boxes[box_col].astype(str).tolist()
        values = dict(zip(strain_boxes[box_col].astype(str), strain_boxes[value_col]))
        subjects_per_box = dict(zip(strain_boxes[box_col].astype(str), strain_boxes['subjects_per_box']))
        
        print(f"Boxes: {boxes}")
        print(f"Values: {values}")
        print(f"Subjects per box: {subjects_per_box}")
        
        try:
            results[strain] = find_optimal_allocation_ilp(boxes, values, n_groups, group_names, subjects_per_box)
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
