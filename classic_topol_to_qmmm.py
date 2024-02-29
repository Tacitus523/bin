import os
import re
import shutil

DATA_FOLDER = "/data/lpetersen/thiol_disulfide/B3LYP_aug-cc-pVTZ_protein/trajectories"
FOLDERS = ["01_OxMo1", "02_ReDi1", "03_ReDi2X7d1", "04_ReMo2X7"] # Folders with trajectories, relative or absolute path
TOPOLS = ["topol.top"]*len(FOLDERS) # Topology files inside FOLDERS, typically called 'topol.top'

MM_LINK_INDEX_LISTS = [[346,1682,1717], [346,2103,2113], [391,346,2102], [3233,346,2103]] # Indices of the MM-Atom to be replaced by a linker atom in the QM-region 
QM_LINK_INDEX_LISTS = [[348,1684,1719], [348,2105,2115], [393,348,2104], [3235,348,2105]] # Indices of the QM-Atom bound to the linker atom in the QM-region
QM_BONDS_LIST = [[[350,348], [349,348], [348,351],
                  [1685,1684], [1686,1684], [1684,1687],
                  [1719,1720], [1719,1721], [1719,1722],
                  [1722,1687]
                  ],
                 [[350,348], [349,348], [348,351],
                  [2105,2106], [2105,2107], [2105,2108],
                  [2115,2116], [2115,2117], [2115,2118],
                  [2108, 2118]
                  ],
                 [[393,394], [393,395], [393,396],
                  [350,348], [349,348], [348,351],
                  [2104,2105], [2104,2106], [2104,2107],
                  [351,2107]
                  ],
                 [[3235,3236], [3235,3237] ,[3235,3238],
                  [350,348], [349,348], [348,351],
                  [2105,2106], [2105,2107], [2105,2108],
                  [351,2108] 
                 ]
                ]

def create_qmmm_topol(data_folder, folders, topols, mm_link_index_lists, qm_link_index_lists, qm_bond_list):
    check_assertions(data_folder, folders, topols, mm_link_index_lists, qm_link_index_lists)
    topols = create_absolute_pathes(data_folder, folders, topols)
    
    # following indices in the list will be incremented, so the list has to be ordered wr.t. the mm index
    mm_qm_index_lists = [sorted(list(zip(mm_indices, qm_indices))) for mm_indices, qm_indices in zip(mm_link_index_lists, qm_link_index_lists)]
    mm_link_index_lists = [[mm_qm_index[0] for mm_qm_index in mm_qm_indices] for mm_qm_indices in mm_qm_index_lists]
    qm_link_index_lists = [[mm_qm_index[1] for mm_qm_index in mm_qm_indices] for mm_qm_indices in mm_qm_index_lists]
    
    zipper = zip(topols, mm_link_index_lists, qm_link_index_lists, qm_bond_list)
    for topol, mm_link_indices, qm_link_indices, qm_bonds in zipper:
        # create_backup_topol(topol)
        with open(topol, "r") as f:
            lines = f.readlines()
        lines = insert_atomtypes(lines)
        lines, la_indices = insert_la_into_molecule(lines, mm_link_indices)
        lines = insert_virtual_sites(lines, mm_link_indices, qm_link_indices, la_indices)
        lines = insert_constraints(lines, mm_link_indices, qm_link_indices)
        lines = alter_bonds(lines, mm_link_indices, qm_link_indices, qm_bonds)
        text = "".join(lines)
        write_changed_topol(topol, text)
        
def insert_la_into_molecule(lines: list, mm_link_indices: list):
    """Puts linker atoms at the appropiate positions in the "atoms" section of the topology

    Args:
        lines (list): Lines in the topology
        mm_link_indices (list): Indices of the positions of the MM-Atom to be replaced by a linker atom in the QM-region
    """
    # patterns to be searched for with regular expressions
    atom_section_pattern = r"^\[ atoms \]" # [ atoms ] at the beginning of the line
    comment_pattern = r"^;" # ; at the beginning of the line
    emptys_line_pattern = r'^\s*$' # Whitespace at the beginning of the line until the end
    
    line_index = 0
    # Runs until the atom section is found
    while True:
        if re.match(atom_section_pattern, lines[line_index]):
            line_index += 1
            check_line_count(line_index, lines)
            break
        else:
            line_index += 1
            check_line_count(line_index, lines)
    
    # Runs until atom section is over and inserts linker atoms at the end
    la_count = len(mm_link_indices)
    la_indices = [] # For storing the indices of the linker atoms
    while True:
        line = lines[line_index]
        # Break if atom section ended
        if re.match(emptys_line_pattern, line):
            for _ in range(la_count):
                final_atom_line = lines[line_index-1]
                la_line, la_index = make_la_line(final_atom_line)
                lines.insert(line_index, la_line) # Insert Linker atom entry
                la_indices.append(la_index)
                line_index += 1
            lines.insert(line_index-la_count, ";link atoms\n")
            line_index += 1
            break
        
        # Skip line if it is a comment
        if re.match(comment_pattern, line):
            line_index += 1
            check_line_count(line_index, lines)
            continue
        
        atom_index_pattern = r"(\d+)\s*"
        match = re.search(atom_index_pattern, line)
        atom_index = int(match.group(1))
        if atom_index in mm_link_indices:
            lines = alter_charges(lines, line_index)
        
        line_index += 1
        check_line_count(line_index, lines)

    return lines, la_indices

def make_la_line(final_atom_line: str):
    #Capture all relevant data
    line_pattern = r"(\d+)\s*(\S+)\s*(\d+)\s*([A-Za-z]+)\s*(\S+)\s*(\d+)\s*([-+]?\d*\.\d+)\s*([-+]?\d+(?:\.\d+)?)" # Capture atom index, atom type, res index, res type, atom type in gro, atom index in gro(?), charge and mass 
    matches = re.search(line_pattern, final_atom_line)
    atom_index = int(matches.group(1))
    atom_type = matches.group(2)
    residue_index = int(matches.group(3))
    residue_type = matches.group(4)
    atom_type_gro = matches.group(5)
    atom_index_gro = int(matches.group(6))
    charge = matches.group(7)
    mass = matches.group(8)
    
    # Replace atom data with linker atom data
    la_line = final_atom_line
    la_line = re.sub(atom_type.rjust(len("LA")), "LA".rjust(len(atom_type)), la_line) # Match and target should have same length, there jusitfy on the right side
    la_line = re.sub(residue_type.rjust(len("XXX")), "XXX".rjust(len(atom_type)), la_line)
    la_line = re.sub(atom_type_gro.rjust(len("LA")), "LA".rjust(len(atom_type_gro)), la_line)
    la_line = re.sub(charge.rjust(len("0.00")), "0.00".rjust(len(charge)), la_line)
    la_line = re.sub(mass.rjust(len("0.00")), "0.00".rjust(len(mass)), la_line)
    la_line = increment_indices(la_line, increment=1)
    comment_pattern = r";.*\n" # comment somwhere in the line
    la_line = re.sub(comment_pattern, "\n", la_line) # replace comment+newline with newline
    la_index = atom_index + 1
    return la_line, la_index
    
def increment_indices(line: str, increment: int):
    """Increments the indices in the [ atoms ] section
    """
    indices_pattern = r"(\s*(\d+)\s*)\S+(\s*(\d+)\s*)[A-Za-z]+\s*\S+(\s*(\d+)\s*)"
    match = re.search(indices_pattern, line)
    old_atom_index, atom_index, old_residue_index, residue_index, old_atom_index_gro, atom_index_gro = match.groups()
    atom_index, residue_index, atom_index_gro = int(atom_index), int(residue_index), int(atom_index_gro)
    new_atom_index = re.sub(str(atom_index).rjust(len(str(atom_index + increment))), str(atom_index + increment), old_atom_index)
    new_residue_index = re.sub(str(residue_index).rjust(len(str(residue_index + increment))), str(residue_index + increment), old_residue_index)
    new_atom_index_gro = re.sub(str(atom_index_gro).rjust(len(str(atom_index_gro + increment))), str(atom_index_gro + increment), old_atom_index_gro)
    line = re.sub(old_atom_index, new_atom_index, line, count=1)
    line = re.sub(old_residue_index, new_residue_index, line, count=1)
    line = re.sub(old_atom_index_gro, new_atom_index_gro, line, count=1)
    return line

def alter_charges(lines: list, line_index: int):
    def format_charges(old_charge: str, new_charge: float):
        new_charge = f"{new_charge:.4f}" if float(old_charge)>0 or new_charge<0 else f" {new_charge:.4f}" # Consider sign change
        old_charge= old_charge.rjust(len(new_charge)) # Consider possible length difference
        return old_charge, new_charge
        
    #Capture all relevant data
    line = lines[line_index]
    line_pattern = r"(\d+)\s*(\S+)\s*(\d+)\s*([A-Za-z]+)\s*(\S+)\s*(\d+)\s*([-+]?\d*\.\d+)\s*([-+]?\d+(?:\.\d+)?)" # Capture atom index, atom type, res index, res type, atom type in gro, atom index in gro(?), charge and mass 
    matches = re.search(line_pattern, line)
    link_residue_index = int(matches.group(3))
    atom_type_gro = matches.group(5)
    charge = matches.group(7)
    
    charge_to_distribute = float(charge)
    
    new_charge = 0.0
    charge, new_charge = format_charges(charge, new_charge)
    line = re.sub(charge, new_charge, line, 1)
    lines[line_index] = line
    
    if atom_type_gro == "CA":
        next_line = lines[line_index+1]
        next_matches = re.search(line_pattern, next_line)
        next_atom_type_gro = next_matches.group(5)
        next_charge = next_matches.group(7)
        if next_atom_type_gro == "HA":
            charge_to_distribute += float(next_charge)
            
            next_new_charge = 0.0
            next_charge, next_new_charge = format_charges(next_charge, next_new_charge)
            next_line = re.sub(next_charge, next_new_charge, next_line, 1)
            lines[line_index+1] = next_line
        else:
            print(f"WARNING: Did not find HA near the MM-Link {line_index}. Only distributing the MM-Link charge")
    else:
        print("WARNING: Script was not intended for linker atom insertion on MM Links other than CA")
    
    # Find beginning of residue
    while True:
        line_index -= 1
        line = lines[line_index]
        matches = re.search(line_pattern, line)
        if matches is None: # No matches means it's probably a comment
            line_index += 1
            break
        current_residue_index = int(matches.group(3))
        if current_residue_index != link_residue_index: # Residue changed?
            line_index += 1
            break
        line_index -= 1
        
    target_atom_types = ["N","H","C","O"]
    charge_per_target = charge_to_distribute/len(target_atom_types)
    # Go through the residue
    while True:
        if len(target_atom_types) == 0: # Break, when all targets are found
            break
        resid__pattern = r"(\d+)\s*(\S+)\s*(\d+)\s*([A-Za-z]+)\s*(\S+)\s*(\d+)\s*([-+]?\d*\.\d+)\s*([-+]?\d+(?:\.\d+)?)"
        line = lines[line_index]
        matches = re.search(line_pattern, line)
        current_residue_index = int(matches.group(3))
        atom_type_gro = matches.group(5)
        charge = matches.group(7)
        
        if current_residue_index != link_residue_index: # Raise an error, when the residue changes.
            raise ValueError(f"Did not find all charge distribution targets near the MM-Link {line_index}. Atom type missing: {target_atom_types}")
        
        if atom_type_gro in target_atom_types:
            new_charge = float(charge)+charge_per_target
            charge, new_charge = format_charges(charge, new_charge)
            line = re.sub(charge, new_charge, line, 1)
            lines[line_index] = line
            target_atom_types.remove(atom_type_gro)
        
        line_index += 1
    return lines

def increment_links_and_bonds(mm_link_indices: list, qm_link_indices: list, bonds: list, la_indices: list):
    """Increment all the indices of the links and the bonds depending on if they are higher
    than the linker atom indices. Therefore they will be correct in the further changes.
    """        
    for la_index in la_indices:
        for i in range(len(mm_link_indices)):
            if mm_link_indices[i] < la_index:
                mm_link_indices[i] += 1
        for i in range(len(qm_link_indices)):
            if qm_link_indices[i] < la_index:
                qm_link_indices[i] += 1
        for i in range(len(bonds)):
            for j in range(2):
                if bonds[i][j] < la_index:
                    bonds[i][j] += 1
    return mm_link_indices, qm_link_indices, bonds

def insert_atomtypes(lines: list):
    line_index = 0
    # Runs until the moleculetype section is found
    moleculetype_section_pattern = r"^\[ moleculetype \]" # [ moleculetype ] at the beginning of the line
    while True:
        if re.match(moleculetype_section_pattern, lines[line_index]):
            break
        else:
            line_index += 1
            check_line_count(line_index, lines)
            
    # Prepare atomtypes section  
    atomtypes_text = """[ atomtypes ]
; name        mass      charge    ptype           sigma             epsilon
    LA   1   0.0000     0.00000        A        0.000000000000      0.00000
    
"""
    lines.insert(line_index, atomtypes_text)
    line_index += 1
    return lines
         
def insert_virtual_sites(lines: list, mm_link_indices: list, qm_link_indices: list, la_indices: list):
    line_index = 0
    # Runs until the bond section is found
    bond_section_pattern = r"^\[ bonds \]" # [ bonds ] at the beginning of the line
    while True:
        if re.match(bond_section_pattern, lines[line_index]):
            break
        else:
            line_index += 1
            check_line_count(line_index, lines)
    
    # Prepare virtual sites section        
    virtual_sites_text = """;define the sites of QM/MM link atoms
[ virtual_sites2 ]
; LA       QM     MM      funct    length
"""
    lines.insert(line_index, virtual_sites_text)
    line_index += 1
    
    # Add virtual sites
    zipper = zip(mm_link_indices, qm_link_indices, la_indices)
    for mm_link_index, qm_link_index, la_index in zipper:
        la_str = str(la_index).rjust(6)
        qm_str = str(qm_link_index).rjust(6)
        mm_str = str(mm_link_index).rjust(6)
        func_str = "1".rjust(6)
        length_str = "0.72".rjust(10)
        virtual_site_str = " ".join([la_str, qm_str, mm_str, func_str, length_str])+"\n"
        lines.insert(line_index, virtual_site_str)
        line_index += 1
    lines.insert(line_index, "\n")
    line_index += 1
    return lines

def insert_constraints(lines: list, mm_link_indices: list, qm_link_indices: list):
    line_index = 0
    # Runs until the bond section is found
    bond_section_pattern = r"^\[ bonds \]" # [ bonds ] at the beginning of the line
    while True:
        if re.match(bond_section_pattern, lines[line_index]):
            break
        else:
            line_index += 1
            check_line_count(line_index, lines)
    
    # Prepare virtual sites section        
    constraints_text = """;constraint the QM-MM bonds
[ constraints ]
;  QM     MM      type      length
"""
    lines.insert(line_index, constraints_text)
    line_index += 1
    
    # Add constraints
    zipper = zip(mm_link_indices, qm_link_indices)
    for mm_link_index, qm_link_index in zipper:
        qm_str = str(qm_link_index).rjust(6)
        mm_str = str(mm_link_index).rjust(6)
        type_str = "2".rjust(6)
        length_str = "0.1538".rjust(10)
        constraint_str = " ".join([qm_str, mm_str, type_str, length_str])+"\n"
        lines.insert(line_index, constraint_str)
        line_index += 1
    lines.insert(line_index, "\n")
    line_index += 1
    return lines
    
def alter_bonds(lines: list, mm_link_indices: list, qm_link_indices: list, qm_bonds: list):
    line_index = 0
    # Runs until the bond section is found
    bond_section_pattern = r"^\[ bonds \]" # [ bonds ] at the beginning of the line
    while True:
        if re.match(bond_section_pattern, lines[line_index]):
            line_index += 1
            check_line_count(line_index, lines)
            break
        else:
            line_index += 1
            check_line_count(line_index, lines)
    bond_section_index = line_index
    
    # Runs until the angles section is found
    angle_section_pattern = r"^\[ angles \]" # [ angles ] at the beginning of the line
    while True:
        if re.match(angle_section_pattern, lines[line_index]):
            line_index += 1
            check_line_count(line_index, lines)
            break
        else:
            line_index += 1
            check_line_count(line_index, lines)
    angle_section_index = line_index
    
    bond_section_text = "ß".join(lines[bond_section_index:angle_section_index]) # use placeholder to join relevant lines
    
    # Comment out QMMM bonds
    zipper = zip(mm_link_indices, qm_link_indices)
    for bond in zipper:
        bond = list(bond)
        bond.sort()
        bond_pattern = "(\s+{}\s+{}\s+\d)".format(bond[0],bond[1])
        bond_pattern = re.compile(bond_pattern)
        match = re.search(bond_pattern, bond_section_text)
        if match is None:
            print(f"Unable to locate QMMM bond {bond}. Aborting")
            exit(1)
        old_qmmm_bond_line = match.group(1)
        new_qmmm_bond_line = ";"+old_qmmm_bond_line
        bond_section_text = re.sub(old_qmmm_bond_line, new_qmmm_bond_line, bond_section_text)
    
    # Alter QM bonds
    for bond in qm_bonds:
        bond = list(bond)
        bond.sort()
        bond_pattern = r"(\s+"+str(bond[0])+r"\s+"+str(bond[1])+r"\s+"+r"\d)"
        match = re.search(bond_pattern, bond_section_text)
        if match is None:
            print(f"Unable to locate QM bond {bond}. Aborting")
            exit(1)
        old_qmmm_bond_line = match.group(1)
        new_qmmm_bond_line = re.sub(r"\d$", "5", old_qmmm_bond_line)
        bond_section_text = re.sub(old_qmmm_bond_line, new_qmmm_bond_line, bond_section_text)
    
    bond_section_lines = bond_section_text.split("ß")
    lines = lines[:bond_section_index] + bond_section_lines + lines[angle_section_index:]
    return lines
         
def create_absolute_pathes(data_folder, *path_parts: list):
    absolute_pathes = []
    zipper = zip(*path_parts)
    for path_parts in zipper:
        absolute_path = os.path.join(data_folder, *path_parts)
        absolute_pathes.append(absolute_path)
    return absolute_pathes

def create_backup_topol(topol: str):
    """To make sure, that correct topols for recovery are still present

    Args:
        topols (str): Absolute path to topology
    """
    backup_name = "topol_original.top"
    parent_directory = os.path.dirname(topol)
    backup_path = os.path.join(parent_directory, backup_name)
    if not os.path.exists(backup_path):
        shutil.copy(topol, backup_path)

def write_changed_topol(topol: str, text: str):
    changed_name = "topol_with_la.top"
    parent_directory = os.path.dirname(topol)
    changed_path = os.path.join(parent_directory, changed_name)
    with open(changed_path, "w") as f:
        f.write(text)
        
def check_assertions(data_folder, folders, topols, mm_link_index_lists, qm_link_index_lists):
    assert all(len(lst) == len(folders) for lst in [folders, topols]), "The folder lists don't have an equal length"
    assert all(len(lst) == len(mm_link_index_lists) for lst in [mm_link_index_lists, qm_link_index_lists]), "The index lists don't have an equal length"
    assert all(len(mm_link_indices) > 0 for mm_link_indices in mm_link_index_lists)
        
def check_line_count(line_index, lines):
    assert line_index < len(lines), "The maximal amount of line has been surpassed without achieving the objective"
    
if __name__=="__main__":
    create_qmmm_topol(DATA_FOLDER, FOLDERS, TOPOLS, MM_LINK_INDEX_LISTS, QM_LINK_INDEX_LISTS, QM_BONDS_LIST)
    


