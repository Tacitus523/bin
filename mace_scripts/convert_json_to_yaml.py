#!/usr/bin/env python3
import json
import yaml
import sys
from pathlib import Path
from typing import Any, Dict

"""Convert JSON file to YAML format."""



def convert_json_to_yaml(json_file: str, yaml_file: str = None) -> None:
    """Convert JSON file to YAML format.
    
    Args:
        json_file: Path to input JSON file
        yaml_file: Path to output YAML file (optional)
    """
    json_path = Path(json_file)
    
    if not json_path.exists():
        print(f"Error: JSON file '{json_file}' not found")
        sys.exit(1)
    
    # Read JSON file
    with open(json_path, "r") as f:
        data: Dict[str, Any] = json.load(f)
    
    # Determine output file
    if yaml_file is None:
        yaml_file = json_path.with_suffix(".yaml")
    
    # Write YAML file
    with open(yaml_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Converted '{json_file}' to '{yaml_file}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_json_to_yaml.py <json_file> [yaml_file]")
        sys.exit(1)
    
    json_input = sys.argv[1]
    yaml_output = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_json_to_yaml(json_input, yaml_output)