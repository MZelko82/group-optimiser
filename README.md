# Group Allocation Optimizer

A Streamlit web application for optimizing group allocations based on numeric values while keeping specified groups together. Perfect for experimental design in research settings where balanced groups are crucial.

## Features

- Upload CSV or Excel files
- Select columns for optimization:
  - Numeric column to balance between groups
  - Group identifier column (e.g., box/cage numbers)
  - Optional strain/type column for separate optimizations
- Interactive visualization of group distributions
- Download results as CSV
- Statistical summaries of group allocations

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd group-optimizer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Open your web browser to the displayed URL (typically http://localhost:8501)

3. Upload your data file (CSV or Excel) with the following columns:
   - A numeric column with values to optimize on (e.g., weights)
   - A column identifying groups that must stay together (e.g., box/cage numbers)
   - Optional: A column for different types if separate optimizations are needed

4. Select the relevant columns:
   - Value column to optimize on (must be numeric)
   - Group identifier column (items that must stay together)
   - Optional: Type column for separate optimizations

5. Click "Optimize Groups" to run the optimization

6. Download the results as a CSV file

## Example Input Format

Your CSV or Excel file should look something like this:

```
ID,Box,Weight,Type
1,Box1,250,A
2,Box1,245,A
3,Box2,260,A
4,Box2,255,A
5,Box3,240,B
6,Box3,235,B
```

Where:
- Box: identifies groups that must stay together
- Weight: the numeric value to optimize on
- Type (optional): for separate optimizations

## Privacy

This application runs locally and does not store any data. All processing happens in-memory and data is cleared when you close the browser or refresh the page.
