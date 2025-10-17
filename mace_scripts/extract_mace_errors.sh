#!/bin/bash

# Script to extract MACE evaluation tables and save as JSON
# Extracts tables in the format shown in training output

set -e

usage() {
    echo "Usage: $0 <output_file> [base_name]"
    echo "  output_file: Path to the training output file"
    echo "  base_name: Base name for output files (default: extracted)"
    echo "  Output: base_name_metrics.json and base_name_swa_metrics.json"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

OUTPUT_FILE="$1"
BASE_NAME="${2:-extracted}"

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Output file $OUTPUT_FILE not found"
    exit 1
fi

# Function to extract values from a table row
extract_values() {
    local line="$1"
    # Split by | and clean whitespace, extract columns 2-5
    echo "$line" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $3); gsub(/^[ \t]+|[ \t]+$/, "", $4); gsub(/^[ \t]+|[ \t]+$/, "", $5); gsub(/^[ \t]+|[ \t]+$/, "", $6); print $3, $4, $5, $6}'
}

# Function to create JSON from table data
create_json() {
    local train_data=($1)
    local valid_data=($2) 
    local test_data=($3)
    local output_file="$4"
    
    cat > "$output_file" << EOF
{
  "train": {
    "rmse_e_per_atom_mev": ${train_data[0]},
    "rmse_f_mev_per_a": ${train_data[1]},
    "rel_rmse_f_percent": ${train_data[2]},
    "rmse_q": ${train_data[3]}
  },
  "valid": {
    "rmse_e_per_atom_mev": ${valid_data[0]},
    "rmse_f_mev_per_a": ${valid_data[1]},
    "rel_rmse_f_percent": ${valid_data[2]},
    "rmse_q": ${valid_data[3]}
  },
  "test": {
    "rmse_e_per_atom_mev": ${test_data[0]},
    "rmse_f_mev_per_a": ${test_data[1]},
    "rel_rmse_f_percent": ${test_data[2]},
    "rmse_q": ${test_data[3]}
  }
}
EOF
}

echo "Extracting tables from $OUTPUT_FILE..."

# Extract first table (before SWA checkpoint loading)
first_table_section=$(awk '
    BEGIN { in_table=0; table_count=0 }
    /config_type.*RMSE E.*RMSE F.*relative F RMSE.*RMSE q/ { in_table=1; print; next }
    in_table && /^\+.*\+$/ { print; next }
    in_table && /^\|.*train.*\|/ { print; next }
    in_table && /^\|.*valid.*\|/ { print; next }  
    in_table && /^\|.*Default.*\|/ { print; next }
    in_table && /Loading checkpoint.*swa/ { in_table=0; exit }
    in_table && /^[^|+]/ { in_table=0; exit }
' "$OUTPUT_FILE")

# Extract second table (after SWA checkpoint loading)
second_table_section=$(awk '
    BEGIN { swa_found=0; in_table=0 }
    /Loading checkpoint.*swa/ { swa_found=1; next }
    swa_found && /config_type.*RMSE E.*RMSE F.*relative F RMSE.*RMSE q/ { in_table=1; print; next }
    swa_found && in_table && /^\+.*\+$/ { print; next }
    swa_found && in_table && /^\|.*train.*\|/ { print; next }
    swa_found && in_table && /^\|.*valid.*\|/ { print; next }  
    swa_found && in_table && /^\|.*Default.*\|/ { print; exit }
' "$OUTPUT_FILE")

# Process first table
if [[ -n "$first_table_section" ]]; then
    echo "Processing first table (regular model)..."
    
    train_line=$(echo "$first_table_section" | grep '|.*train.*|')
    valid_line=$(echo "$first_table_section" | grep '|.*valid.*|')
    test_line=$(echo "$first_table_section" | grep '|.*Default.*|')
    
    if [[ -n "$train_line" && -n "$valid_line" && -n "$test_line" ]]; then
        train_values=$(extract_values "$train_line")
        valid_values=$(extract_values "$valid_line")
        test_values=$(extract_values "$test_line")
        
        create_json "$train_values" "$valid_values" "$test_values" "${BASE_NAME}_metrics.json"
        echo "Saved regular model metrics to ${BASE_NAME}_metrics.json"
        
        echo "Regular model values:"
        echo "  Train: $train_values"
        echo "  Valid: $valid_values"
        echo "  Test:  $test_values"
    else
        echo "Warning: Could not parse first table completely"
    fi
else
    echo "Warning: Could not find first table"
fi

# Process second table
if [[ -n "$second_table_section" ]]; then
    echo "Processing second table (SWA model)..."
    
    train_line=$(echo "$second_table_section" | grep '|.*train.*|')
    valid_line=$(echo "$second_table_section" | grep '|.*valid.*|')
    test_line=$(echo "$second_table_section" | grep '|.*Default.*|')
    
    if [[ -n "$train_line" && -n "$valid_line" && -n "$test_line" ]]; then
        train_values=$(extract_values "$train_line")
        valid_values=$(extract_values "$valid_line")
        test_values=$(extract_values "$test_line")
        
        create_json "$train_values" "$valid_values" "$test_values" "${BASE_NAME}_swa_metrics.json"
        echo "Saved SWA model metrics to ${BASE_NAME}_swa_metrics.json"
        
        echo "SWA model values:"
        echo "  Train: $train_values"
        echo "  Valid: $valid_values"
        echo "  Test:  $test_values"
    else
        echo "Warning: Could not parse second table completely"
    fi
else
    echo "Warning: Could not find second table"
fi

echo "Done!"