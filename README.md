# Group Allocation Optimizer

A web application for optimizing experimental group allocations while maintaining box/cage integrity. Uses Integer Linear Programming (ILP) to create balanced groups based on numeric values (e.g., weights) while ensuring animals from the same box stay together.

## Primary Access

The app is hosted on Streamlit Cloud and can be accessed at:

[https://group-optimiser-wcc2dtwleonkb38i3vyxfu.streamlit.app/](https://group-optimiser-wcc2dtwleonkb38i3vyxfu.streamlit.app/)

## Features

- Upload CSV files with experimental data
- Optimize group allocations based on:
  - Numeric values (e.g., weights) to balance between groups
  - Box/cage identifiers to maintain group integrity
  - Optional grouping variable (e.g., strain, age) for stratified allocation
- Interactive visualizations:
  - Initial distribution plots
  - Optimized group distribution plots
  - Combined distribution plots when using grouping variables
- Statistical summaries by group and subgroup
- Customizable group labels
- Download optimized allocations as CSV

## Input Requirements

Your CSV file should contain:

1. **Required Columns**:
   - A numeric column (e.g., weights, measurements) to balance between groups
   - A box/cage identifier column (e.g., box numbers, cage IDs) for subjects that must stay together

2. **Optional Column**:
   - A grouping variable (e.g., strain, age) if you need stratified allocation

Example CSV format:
```csv
ID,Box,Weight,Strain
1,Box1,250,WT
2,Box1,245,WT
3,Box2,260,WT
4,Box2,255,WT
5,Box3,240,KO
6,Box3,235,KO
```

## Local Installation (Optional)

If you prefer to run the app locally:

1. Clone this repository:
```bash
git clone https://github.com/MZelko82/group-optimiser.git
cd group-optimizer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run streamlit_app.py
```

## Usage

1. Upload your CSV file
2. Select the columns for optimization:
   - Value column to optimize (must be numeric)
   - Box/cage identifier column
   - Optional grouping variable column
3. Configure groups:
   - Set number of groups (2-10)
   - Customize group labels if desired
4. Click "Optimize Groups" to run
5. Review the results:
   - Initial distribution plot
   - Group summary statistics
   - Optimized distribution plots
6. Download the results as CSV

## Privacy

This application processes all data in-memory and does not store any information permanently. Data is cleared when you close the browser or refresh the page. When using the hosted version, data is transmitted securely via HTTPS.
