import os
from pbix_mcp.builder import PBIXBuilder

print("Testing PBIXBuilder...")
builder = PBIXBuilder("TestModel")

# Add a simple table
columns = [
    {"name": "ID", "data_type": "Int64"},
    {"name": "Name", "data_type": "String"},
    {"name": "Value", "data_type": "Double"}
]
rows = [
    {"ID": 1, "Name": "Alice", "Value": 10.5},
    {"ID": 2, "Name": "Bob", "Value": 20.3}
]
builder.add_table("Users", columns, rows)

# Add a measure
builder.add_measure("Users", "Total Value", "SUM('Users'[Value])")

# Add a page with a card visual
visuals = [
    {
        "type": "card",
        "name": "total_val_card",
        "config": {"measure": "Total Value"},
        "x": 50,
        "y": 50,
        "width": 200,
        "height": 100
    }
]
builder.add_page("Overview", visuals)

# Save the file
output_path = "test_output.pbix"
builder.save(output_path)
print(f"File saved successfully to {output_path}. Size: {os.path.getsize(output_path)} bytes")
