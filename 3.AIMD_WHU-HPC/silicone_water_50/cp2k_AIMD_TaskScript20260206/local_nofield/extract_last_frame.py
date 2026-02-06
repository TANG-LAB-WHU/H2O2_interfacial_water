#!/usr/bin/env python
"""Extract the last frame from a CP2K trajectory XYZ file."""

import sys
from pathlib import Path


def extract_last_frame(input_file: str, output_file: str) -> None:
    """Extract the last frame from a trajectory XYZ file.
    
    Args:
        input_file: Path to input trajectory file
        output_file: Path to output file for last frame
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)
    
    # Read all lines
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Total lines in trajectory: {total_lines}")
    
    # Get atom count from first line of last frame
    # XYZ format: first line is atom count, second is comment, rest are coordinates
    # We need to find the last frame by reading atom count
    
    # Read atom count from the first line
    atom_count = int(lines[0].strip())
    frame_size = atom_count + 2  # atoms + header (2 lines)
    
    print(f"Atom count: {atom_count}")
    print(f"Frame size: {frame_size} lines")
    
    # Calculate last frame start position
    last_frame_start = total_lines - frame_size
    
    if last_frame_start < 0:
        print("ERROR: Could not find valid frame in trajectory")
        sys.exit(1)
    
    print(f"Last frame starts at line: {last_frame_start + 1}")
    
    # Extract last frame
    last_frame = lines[last_frame_start:]
    
    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(last_frame)
    
    # Parse energy from comment line
    comment_line = last_frame[1].strip()
    print(f"Comment: {comment_line}")
    print(f"SUCCESS: Extracted last frame with {atom_count} atoms to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_last_frame.py <input_trajectory> <output_xyz>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    extract_last_frame(input_file, output_file)
