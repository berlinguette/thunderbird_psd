from pathlib import Path
from typing import Optional

def main(specfile_name: str):
    specfile_path = Path(__file__).parent / specfile_name
    
    with open(specfile_path, "r") as specfile:
        lines = specfile.readlines()
        print(f"Specfile read ({len(lines)} lines)")

    index: Optional[int] = None
    for i, line in enumerate(lines):
        if 'text_pos' in line:
            index = i
            break
    if index is not None:
        print(f"Found text_pos on line {index}, editing")
        pos_line = lines[index].replace('None', '(10,420)')
        lines[index] = pos_line
        print(f"Adding text_color at line {index+1}")
        lines.insert(index+1, "    text_color='white',\n")
                    
    with open(specfile_path, 'w') as specfile:
        specfile.writelines(lines)
        print(f"Specfile edits complete")

if __name__ == "__main__":
    main('neutron_data_converter.spec')