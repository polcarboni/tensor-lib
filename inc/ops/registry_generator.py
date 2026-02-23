import re
from pathlib import Path


def get_structs_from_header(filepath: Path) -> list[str]:
    '''Extract top-level struct/class named from a header file.'''
    print("_getstructsform header()_")
    content = filepath.read_text()
    # print(content)
    pattern = r'^\s*(?:struct|class)\s+(\w+)'
    structs = re.findall(pattern, content, re.MULTILINE)
    print(structs)
    forward = [s for s in structs if "Backward" not in s]
    backward = [s for s in structs if "Backward" in s]
    
    return forward, backward


def camel_to_snake_upper(name: str) -> str:
    snake = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', name).upper()
    return snake


def make_macro_block(macro_name: str, structs: list[str], header_name: str | None) -> str:
    '''Builds a #define X(macro) block, optionally preceded by an #include.'''
    lines = []
    if header_name:
        lines.append(f'#include"{header_name}"')
    lines.append(f"#define {macro_name}(X)    \\")
    for i, struct in enumerate(structs):
        is_last = (i == len(structs) - 1)
        continuation = "" if is_last else "    \\"
        lines.append(f"    X({struct}){continuation}")
    return "\n".join(lines)
    
    
def generate_ops_file(directory: Path, output_file: Path):
    headers = sorted(directory.glob("*Ops.hpp"))
    
    sections = []
    all_group_macros = []
    
    for header in headers:
        forward, backward = get_structs_from_header(header)
        if not forward and backward:
            continue
        stem = header.stem
        snake = camel_to_snake_upper(stem)
        fwd_macro = f"FOR_EACH_{snake}"
        block_lines = []
        
        if forward:
            block_lines.append(make_macro_block(fwd_macro, forward, header.name))
            all_group_macros.append(fwd_macro)
        
        if backward:
            bwd_macro = f"FOR_EACH_{snake}_BACKWARD"
            block_lines.append(make_macro_block(bwd_macro, backward, None))
            all_group_macros.append(bwd_macro)
    
        sections.append("\n\n".join(block_lines))
    
    combiner_lines = ["#define FOR_EACH_OP(X)    \\"]
    for i, name in enumerate(all_group_macros):
        is_last = (i == len(all_group_macros) - 1)
        continuation = "" if is_last else "    \\"
        combiner_lines.append(f"    {name}(X){continuation}")
    
    output_lines = ["#pragma once", ""]
    for section in sections:
        output_lines.append(section)
        output_lines.append("")
    output_lines.append("\n".join(combiner_lines))
    output_lines.append("")
    
    output_file.write_text("\n".join(output_lines))
    
if __name__ == "__main__":
    import sys
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else directory / "OPOPOPS.hpp"
    generate_ops_file(directory, output)