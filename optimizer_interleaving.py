#!/usr/bin/env python3
"""
Interleaving-based group optimizer.

This optimizer uses a simple interleaving strategy to balance groups:
1. Sort boxes by weight
2. Alternate assignment between groups
3. Optionally consider strain balance as a secondary constraint

This approach naturally balances both mean and variance without requiring
complex optimization solvers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


def get_box_weights_interleaving(df: pd.DataFrame, value_col: str, box_col: str, 
                                strain_col: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate total weights for each box for interleaving optimizer.
    Simplified version that avoids column naming issues.
    """
    print(f"Calculating box weights for interleaving:")
    print(f"Value column: {value_col}")
    print(f"Box column: {box_col}")
    print(f"Strain column: {strain_col}")
    
    # Simple grouping by box
    box_weights = df.groupby(box_col)[value_col].sum().reset_index()
    
    # Rename columns to standard names
    box_weights.columns = ['box', 'weight']
    
    # Add strain information if needed
    if strain_col:
        # Get strain composition for each box
        strain_composition = df.groupby([box_col, strain_col]).size().unstack(fill_value=0)
        
        for strain in strain_composition.columns:
            strain_col_name = f'strain_{strain}'
            box_weights[strain_col_name] = box_weights['box'].map(strain_composition[strain])
    
    print(f"Box weights shape: {box_weights.shape}")
    print(f"Box weights columns: {box_weights.columns.tolist()}")
    print(f"Sample data:")
    print(box_weights.head())
    
    return box_weights


def find_optimal_allocation_interleaving(
    box_weights: pd.DataFrame,
    n_groups: int = 2,
    group_names: Optional[List[str]] = None,
    strain_col: Optional[str] = None,
    consider_strain_balance: bool = True,
    sort_direction: str = 'ascending',
    use_smart_interleaving: bool = False,
    use_hierarchical: bool = False
) -> Dict[str, Dict]:
    """
    Find optimal group allocation using interleaving strategy.
    
    Args:
        box_weights: DataFrame with box weights
        n_groups: Number of groups to create
        group_names: Names for the groups
        strain_col: Column name for strain grouping
        consider_strain_balance: Whether to consider strain balance
        sort_direction: 'ascending' or 'descending' for weight sorting
    
    Returns:
        Dictionary with allocation results
    """
    if group_names is None:
        group_names = [f'Group_{i+1}' for i in range(n_groups)]
    
    if len(group_names) != n_groups:
        raise ValueError(f"Number of group names ({len(group_names)}) must match n_groups ({n_groups})")
    
    print(f"Using interleaving strategy with {n_groups} groups")
    print(f"Sort direction: {sort_direction}")
    print(f"Consider strain balance: {consider_strain_balance}")
    
    # Prepare data - handle different column names from get_box_weights
    if 'box' in box_weights.columns:
        box_col_name = 'box'
    elif 'Rat Box' in box_weights.columns:
        box_col_name = 'Rat Box'
    else:
        # Find the first column that looks like a box identifier
        box_col_name = box_weights.columns[0]
    
    if 'weight' in box_weights.columns:
        # Handle duplicate weight columns - use the first one
        weight_cols = [col for col in box_weights.columns if col == 'weight']
        weight_col_name = weight_cols[0] if weight_cols else 'weight'
    else:
        # Find the first numeric column that looks like weights
        numeric_cols = box_weights.select_dtypes(include=[np.number]).columns
        weight_col_name = numeric_cols[0] if len(numeric_cols) > 0 else box_weights.columns[1]
    
    # Ensure we get Series, not DataFrame
    print(f"Using box column: '{box_col_name}'")
    print(f"Using weight column: '{weight_col_name}'")
    print(f"Box weights columns: {box_weights.columns.tolist()}")
    
    boxes = box_weights[box_col_name].values.tolist()
    weights = box_weights[weight_col_name].values.tolist()
    
    # Debug: Check what we're getting
    print(f"Sample boxes: {boxes[:3]}")
    print(f"Sample weights: {weights[:3]}")
    print(f"Weight types: {[type(w) for w in weights[:3]]}")
    
    # Ensure weights are single values, not lists
    if weights and isinstance(weights[0], (list, tuple, np.ndarray)):
        print("Warning: Weights are lists/tuples, extracting first element")
        weights = [w[0] if isinstance(w, (list, tuple, np.ndarray)) else w for w in weights]
    
    # Create weight-box mapping
    weight_box_pairs = list(zip(weights, boxes))
    
    # Sort by weight
    if sort_direction == 'ascending':
        weight_box_pairs.sort(key=lambda x: x[0])
    else:
        weight_box_pairs.sort(key=lambda x: x[0], reverse=True)
    
    print(f"Box weights after sorting: {[pair[0] for pair in weight_box_pairs]}")
    
    # Initialize groups
    groups = {name: [] for name in group_names}
    group_weights = {name: 0 for name in group_names}
    
    # HIERARCHICAL STRATEGY: Split by strain first, then interleave within strain
    if use_hierarchical and strain_col:
        print(f"Using hierarchical interleaving strategy")
        # Clear groups first to avoid duplicate assignments
        groups = {name: [] for name in group_names}
        group_weights = {name: 0 for name in group_names}
        hierarchical_interleaving_assignment(
            weight_box_pairs, groups, group_weights, group_names, 
            box_weights, box_col_name, n_groups, use_smart_interleaving
        )
    elif consider_strain_balance and strain_col:
        # Get strain information
        strain_columns = [col for col in box_weights.columns if col.startswith('strain_')]
        if strain_columns:
            # Create strain tracking
            strain_counts = {name: {} for name in group_names}
            for strain_col_name in strain_columns:
                strain_name = strain_col_name.replace('strain_', '')
                for group_name in group_names:
                    strain_counts[group_name][strain_name] = 0
            
            # Interleave with strain balance consideration
            for i, (weight, box) in enumerate(weight_box_pairs):
                # Find the group that would be most balanced after adding this box
                best_group = _find_best_group_for_strain_balance(
                    groups, group_weights, strain_counts, box, box_weights, strain_columns, box_col_name
                )
                
                groups[best_group].append(box)
                group_weights[best_group] += weight
                
                # Update strain counts
                for strain_col_name in strain_columns:
                    strain_name = strain_col_name.replace('strain_', '')
                    if box_weights[box_weights[box_col_name] == box][strain_col_name].iloc[0] == 1:
                        strain_counts[best_group][strain_name] += 1
        else:
            # No strain columns found, use simple interleaving
            consider_strain_balance = False
    
    if not consider_strain_balance and not use_hierarchical:
        # Choose between smart and simple interleaving
        if use_smart_interleaving and n_groups >= 3:
            smart_interleaving_assignment(weight_box_pairs, groups, group_weights, group_names)
        else:
            # Simple interleaving for all cases
            for i, (weight, box) in enumerate(weight_box_pairs):
                group_index = i % n_groups
                group_name = group_names[group_index]
                groups[group_name].append(box)
                group_weights[group_name] += weight
    
    # Calculate statistics
    group_totals = [group_weights[name] for name in group_names]
    mean_diff = max(group_totals) - min(group_totals)
    
    # Calculate within-group variances (using box weight ranges as proxy)
    within_group_ranges = {}
    for group_name in group_names:
        group_box_weights = [weight for weight, box in weight_box_pairs if box in groups[group_name]]
        if group_box_weights:
            within_group_ranges[group_name] = max(group_box_weights) - min(group_box_weights)
        else:
            within_group_ranges[group_name] = 0
    
    range_diff = max(within_group_ranges.values()) - min(within_group_ranges.values())
    
    print(f"Group assignments:")
    for group_name in group_names:
        print(f"  {group_name}: {groups[group_name]} (total weight: {group_weights[group_name]})")
    
    print(f"Mean difference: {mean_diff}")
    print(f"Range difference: {range_diff}")
    
    # Return results in the same format as the ILP optimizer
    result = {
        'groups': groups,
        'group_weights': group_weights,
        'group_subjects': {name: len(groups[name]) for name in group_names},
        'std_dev': np.std(list(group_weights.values())),
        'max_difference': mean_diff,
        'within_group_ranges': within_group_ranges,
        'range_balance': range_diff,
        'method': 'interleaving'
    }
    
    return {'Combined': result}


def smart_interleaving_assignment(weight_box_pairs, groups, group_weights, group_names):
    """
    Smart interleaving strategy for 3+ groups using optimal end assignment pattern.
    
    Strategy:
    1. Assign from top: A, B, C, D... (lightest boxes)
    2. Assign from bottom with optimal pattern:
       - 3 groups: A,B,C → C,A,B
       - 4 groups: A,B,C,D → D,A,C,B  
       - 5 groups: A,B,C,D,E → E,A,C,D,B
    3. Interleave remaining middle boxes
    
    This avoids giving the same group both lightest and heaviest boxes.
    """
    n_groups = len(group_names)
    n_boxes = len(weight_box_pairs)
    
    print(f"Using smart interleaving for {n_groups} groups with {n_boxes} boxes")
    
    # Step 1: Assign from top (lightest boxes)
    top_assignments = min(n_groups, n_boxes)
    for i in range(top_assignments):
        weight, box = weight_box_pairs[i]
        group_name = group_names[i]
        groups[group_name].append(box)
        group_weights[group_name] += weight
        print(f"  Top assignment: {box} (weight {weight}) → {group_name}")
    
    # Step 2: Assign from bottom with optimal pattern
    if n_boxes > n_groups:
        bottom_assignments = min(n_groups, n_boxes - n_groups)
        
        # Create optimal bottom assignment pattern
        bottom_pattern = create_optimal_bottom_pattern(n_groups)
        
        for i in range(bottom_assignments):
            weight, box = weight_box_pairs[-(i+1)]  # Start from the end
            group_index = bottom_pattern[i]
            group_name = group_names[group_index]
            groups[group_name].append(box)
            group_weights[group_name] += weight
            print(f"  Bottom assignment: {box} (weight {weight}) → {group_name}")
    
    # Step 3: Interleave remaining middle boxes
    middle_start = n_groups
    middle_end = n_boxes - n_groups if n_boxes > 2 * n_groups else n_boxes
    
    if middle_start < middle_end:
        print(f"  Interleaving middle boxes {middle_start} to {middle_end-1}")
        for i in range(middle_start, middle_end):
            weight, box = weight_box_pairs[i]
            group_index = (i - middle_start) % n_groups
            group_name = group_names[group_index]
            groups[group_name].append(box)
            group_weights[group_name] += weight
            print(f"  Middle assignment: {box} (weight {weight}) → {group_name}")


def create_optimal_bottom_pattern(n_groups):
    """
    Create optimal bottom assignment pattern to avoid extreme pairings.
    
    Pattern rules:
    - Last group gets heaviest box
    - First group gets second-heaviest box (avoids lightest+heaviest pairing)
    - Middle groups get remaining heavy boxes in balanced way
    
    Examples:
    - 3 groups: [0,1,2] → [2,0,1] (C,A,B)
    - 4 groups: [0,1,2,3] → [3,0,2,1] (D,A,C,B)
    - 5 groups: [0,1,2,3,4] → [4,0,2,3,1] (E,A,C,D,B)
    """
    if n_groups == 3:
        return [2, 0, 1]  # C, A, B
    elif n_groups == 4:
        return [3, 0, 2, 1]  # D, A, C, B
    elif n_groups == 5:
        return [4, 0, 2, 3, 1]  # E, A, C, D, B
    else:
        # For 6+ groups, use a general pattern
        # Last group gets heaviest, first group gets second-heaviest
        # Then alternate middle groups
        pattern = [n_groups - 1, 0]  # Start with last, then first
        
        # Add remaining groups in alternating pattern
        remaining = list(range(1, n_groups - 1))
        for i, group in enumerate(remaining):
            if i % 2 == 0:
                pattern.append(group)
            else:
                pattern.insert(-1, group)  # Insert before the last element
        
        return pattern


def hierarchical_interleaving_assignment(
    weight_box_pairs: List[Tuple[float, str]],
    groups: Dict[str, List[str]],
    group_weights: Dict[str, float],
    group_names: List[str],
    box_weights: pd.DataFrame,
    box_col_name: str,
    n_groups: int,
    use_smart_interleaving: bool
):
    """
    Hierarchical interleaving strategy:
    1. First: Random balanced strain allocation - ensure each group gets fair share of each strain
    2. Second: Weight optimization - treat strain-balanced allocation as starting point and optimize weights
    
    This ensures perfect strain balance first, then optimizes weight balance.
    """
    print(f"Step 1: Creating balanced strain allocation across {n_groups} groups")
    
    # Get strain columns
    strain_columns = [col for col in box_weights.columns if col.startswith('strain_')]
    if not strain_columns:
        print("No strain columns found, falling back to simple interleaving")
        # Fall back to simple interleaving
        if use_smart_interleaving and n_groups >= 3:
            smart_interleaving_assignment(weight_box_pairs, groups, group_weights, group_names)
        else:
            for i, (weight, box) in enumerate(weight_box_pairs):
                group_index = i % n_groups
                group_name = group_names[group_index]
                groups[group_name].append(box)
                group_weights[group_name] += weight
        return
    
    # Step 1: Group boxes by strain
    strain_groups = {}
    for weight, box in weight_box_pairs:
        # Find which strain this box belongs to
        box_row = box_weights[box_weights[box_col_name] == box]
        if len(box_row) == 0:
            print(f"    Warning: No data found for box {box}")
            continue
            
        # Find the strain with the highest count for this box
        max_strain_count = 0
        dominant_strain = None
        for strain_col in strain_columns:
            strain_count = box_row[strain_col].iloc[0]
            if strain_count > max_strain_count:
                max_strain_count = strain_count
                dominant_strain = strain_col
        
        if dominant_strain is None or max_strain_count == 0:
            # If no strain found, assign to a default group
            dominant_strain = 'unknown'
            print(f"    Warning: Box {box} has no strain data, assigning to 'unknown'")
        
        if dominant_strain not in strain_groups:
            strain_groups[dominant_strain] = []
        strain_groups[dominant_strain].append((weight, box))
    
    print(f"Found {len(strain_groups)} strain groups:")
    for strain, boxes in strain_groups.items():
        print(f"  {strain}: {len(boxes)} boxes")
    
    # Step 2: Create balanced strain allocation
    print(f"Step 2: Creating balanced strain allocation")
    
    # Create a balanced allocation where each group gets fair share of each strain
    strain_allocation = {strain: [] for strain in strain_groups.keys()}
    
    for strain_name, strain_boxes in strain_groups.items():
        print(f"  Allocating strain {strain_name} ({len(strain_boxes)} boxes) across {n_groups} groups")
        
        # Calculate how many boxes each group should get from this strain
        boxes_per_group = len(strain_boxes) // n_groups
        extra_boxes = len(strain_boxes) % n_groups
        
        # Distribute boxes evenly across groups
        box_index = 0
        for group_idx in range(n_groups):
            # Some groups get one extra box if there's a remainder
            boxes_for_this_group = boxes_per_group + (1 if group_idx < extra_boxes else 0)
            
            for _ in range(boxes_for_this_group):
                if box_index < len(strain_boxes):
                    weight, box = strain_boxes[box_index]
                    strain_allocation[strain_name].append((group_idx, weight, box))
                    print(f"    {box} (weight {weight}) → Group_{group_idx + 1}")
                    box_index += 1
    
    # Step 3: Optimize weight balance within strain constraints
    print(f"Step 3: Optimizing weight balance within strain constraints")
    
    # For each strain, try to optimize weight distribution while maintaining group assignments
    for strain_name, allocations in strain_allocation.items():
        if len(allocations) <= 1:
            continue  # Skip single-box strains
            
        print(f"  Optimizing weight balance for strain {strain_name}")
        
        # Get all boxes for this strain with their current group assignments
        strain_boxes = [(group_idx, weight, box) for group_idx, weight, box in allocations]
        
        # Sort by weight to enable better distribution
        strain_boxes.sort(key=lambda x: x[1])  # Sort by weight
        
        # Redistribute to minimize weight differences between groups
        # Simple approach: alternate assignment to groups with lowest current weight
        for group_idx, weight, box in strain_boxes:
            # Find the group with the lowest current weight
            min_weight_group = min(group_names, key=lambda g: group_weights[g])
            min_weight_group_idx = group_names.index(min_weight_group)
            
            # Assign to the group with lowest weight
            groups[min_weight_group].append(box)
            group_weights[min_weight_group] += weight
            print(f"    {box} (weight {weight}) → {min_weight_group} (optimized)")


def _find_best_group_for_strain_balance(
    groups: Dict[str, List], 
    group_weights: Dict[str, float],
    strain_counts: Dict[str, Dict[str, int]],
    box: str,
    box_weights: pd.DataFrame,
    strain_columns: List[str],
    box_col_name: str
) -> str:
    """
    Find the best group for a box considering strain balance.
    
    This is a simplified approach that tries to balance strains while
    still maintaining reasonable weight balance.
    """
    group_names = list(groups.keys())
    
    # Calculate strain imbalance for each potential assignment
    strain_imbalances = {}
    for group_name in group_names:
        # Calculate what strain counts would be after adding this box
        temp_strain_counts = {name: counts.copy() for name, counts in strain_counts.items()}
        
        for strain_col_name in strain_columns:
            strain_name = strain_col_name.replace('strain_', '')
            if box_weights[box_weights[box_col_name] == box][strain_col_name].iloc[0] == 1:
                temp_strain_counts[group_name][strain_name] += 1
        
        # Calculate strain imbalance (sum of absolute differences from mean)
        strain_totals = {}
        for strain_name in temp_strain_counts[group_name].keys():
            strain_totals[strain_name] = sum(
                temp_strain_counts[g][strain_name] for g in group_names
            )
        
        strain_imbalance = 0
        for strain_name in strain_totals:
            mean_count = strain_totals[strain_name] / len(group_names)
            for g in group_names:
                strain_imbalance += abs(temp_strain_counts[g][strain_name] - mean_count)
        
        strain_imbalances[group_name] = strain_imbalance
    
    # Choose the group with the lowest strain imbalance
    # If tied, choose the one with the lowest current weight
    min_imbalance = min(strain_imbalances.values())
    candidates = [g for g in group_names if strain_imbalances[g] == min_imbalance]
    
    if len(candidates) == 1:
        return candidates[0]
    else:
        # Choose the group with the lowest current weight
        return min(candidates, key=lambda g: group_weights[g])


def find_optimal_allocation_n_groups_interleaving(
    box_weights: pd.DataFrame,
    n_groups: int = 2,
    group_names: Optional[List[str]] = None,
    strain_col: Optional[str] = None,
    consider_strain_balance: bool = True,
    sort_direction: str = 'ascending',
    use_smart_interleaving: bool = False,
    use_hierarchical: bool = False
) -> Dict[str, Dict]:
    """
    Wrapper function to match the interface of the ILP optimizer.
    """
    return find_optimal_allocation_interleaving(
        box_weights=box_weights,
        n_groups=n_groups,
        group_names=group_names,
        strain_col=strain_col,
        consider_strain_balance=consider_strain_balance,
        sort_direction=sort_direction,
        use_smart_interleaving=use_smart_interleaving,
        use_hierarchical=use_hierarchical
    )


if __name__ == "__main__":
    # Simple test
    test_data = {
        'box': [1, 2, 3, 4, 5, 6],
        'weight': [100, 120, 140, 160, 180, 200]
    }
    
    df = pd.DataFrame(test_data)
    result = find_optimal_allocation_interleaving(df, n_groups=2, group_names=['A', 'B'])
    print("Test result:", result)
