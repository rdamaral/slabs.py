import subprocess
import importlib

def check_dependency(dependency):
    try:
        importlib.import_module(dependency)
    except ImportError:
        print(f"{dependency} is not installed. Installing...")
        install_dependency(dependency)

def install_dependency(dependency):
    subprocess.check_call(["pip", "install", dependency])
    print(f"{dependency} has been installed successfully.")

# Check and install dependencies
check_dependency("matplotlib")
check_dependency("pymatgen")

import re
import warnings
import sys
from matplotlib import pyplot as plt
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure
from pymatgen.core.surface import *
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.adsorption import *
from pymatgen.transformations.standard_transformations import AutoOxiStateDecorationTransformation
from pymatgen.transformations.standard_transformations import RotationTransformation
from collections import defaultdict

warnings.simplefilter("ignore", UserWarning)

header = '''
    The Pennsylvania State University
    ╔═╗┬─┐┌─┐┌─┐  ╔╦╗┌─┐┌─┐┌┬┐┌─┐|┌─┐⠀⠀⠀
    ╠═╝├┬┘│ │├┤    ║║┌─┘├─┤ ││├┤  └─┐
    ╩  ┴└─└─┘└    ═╩╝└─┘┴ ┴─┴┘└─┘ └─┘
       ╔╦╗╔╦╗╔╦╗  ╔═╗┬─┐┌─┐┬ ┬┌─┐
       ║║║║║║ ║   ║ ╦├┬┘│ ││ │├─┘
       ╩ ╩╩ ╩ ╩   ╚═╝┴└─└─┘└─┘┴

slabs.py (07.17.2023)     Ricardo Amaral
########################################'''

def convert_poscar_coords(filename):
    # Read the POSCAR file
    poscar = Poscar.from_file(filename)

    # Convert the fractional coordinates to cartesian coordinates
    cart_coords = poscar.structure.lattice.get_cartesian_coords(poscar.structure.frac_coords)

    # Read the POSCAR file into a list of lines
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find the line index where the coordinates start
    coord_start = None
    for i, line in enumerate(lines):
        if 'irect' in line or 'artesian' in line:
            coord_start = i + 1
            break

    # Check if the coordinate type line was found
    if coord_start is None:
        raise ValueError('Coordinate type not found in POSCAR file')

    # Replace the coordinate lines with the new cartesian coordinates
    for i, coord in enumerate(cart_coords):
        # Split the original coordinate line into fields
        fields = lines[coord_start + i].split()

        # Replace the first three fields with the new cartesian coordinates
        fields[:3] = ['  {:19.16f}'.format(x) for x in coord]

        # Join the fields back into a line and add a newline character
        lines[coord_start + i] = ' '.join(fields) + '\n'

    # Replace the coordinate type with 'Cartesian'
    lines[coord_start - 1] = 'Cartesian\n'

    # Write the new POSCAR file
    with open(filename + '2', 'w') as f:
        f.writelines(lines)

def compile_elements(n,t1,t2,slab):

    c = slab.lattice.matrix[2][2]
    shift = get_slab_regions(slab, blength=3.5)[0][0]
    thickness = c*(get_slab_regions(slab, blength=3.5)[0][1]-get_slab_regions(slab, blength=3.5)[0][0])
    vacuum = c*(1-get_slab_regions(slab, blength=3.5)[0][1])

    asf = AdsorbateSiteFinder(slab)
    ads_sites = asf.find_adsorption_sites()

    frac_ads_sites = {}
    for site_type, cart_coords in ads_sites.items():
        frac_coords = [slab.lattice.get_fractional_coords(site) for site in cart_coords]
        frac_ads_sites[site_type] = frac_coords

    inverse_slab = Structure.from_sites(slab)

    theta = np.radians(180)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(theta), -np.sin(theta)],
                                [0, np.sin(theta), np.cos(theta)]])

    for site in inverse_slab:
        rotated_coords = np.dot(rotation_matrix, site.coords)
        site.coords = rotated_coords

    all_indices = [i for i, site in enumerate(inverse_slab)]
    inverse_slab.translate_sites(all_indices, [0, 0, -(vacuum/c)])

    inverse_asf = AdsorbateSiteFinder(inverse_slab)
    inverse_ads_sites = inverse_asf.find_adsorption_sites()

    inverse_frac_ads_sites = {}
    for site_type, cart_coords in inverse_ads_sites.items():
        inverse_frac_coords = [inverse_slab.lattice.get_fractional_coords(inverse_site) for inverse_site in cart_coords]
        inverse_frac_ads_sites[site_type] = inverse_frac_coords

    # shift slab to z = 0
    all_indices = [i for i, site in enumerate(slab)]
    slab.translate_sites(all_indices, [0, 0, -shift])

    # change vacuum size to script input
    # new_lattice_matrix = slab.lattice.matrix.copy()
    # new_lattice_matrix[2][2] = get_slab_regions(slab, blength=3.5)[0][1]*c + 15
    # slab.lattice = Lattice(new_lattice_matrix)

    data = str(slab).strip().split('\n')[9:]
    elements = defaultdict(list)

    for line in data:
        parts = line.split()
        key = float(parts[4])
        element = parts[1]
        elements[key].append(element)

    elements = sorted(elements.items(), reverse=True)
    layers = len(elements)
    composition_ref = re.findall(r'\d+', structure_oxi.composition.formula)
    composition_slab = re.findall(r'\d+', slab.composition.formula)
    composition = ''.join(str(slab.composition.formula).split(' ')) + ", " + ':'.join(''.join(match) for match in re.findall(r'(\D+)(\d?+)', slab.composition.reduced_formula))

    # print(slab.get_surface_sites())

    compiled_list = [f"Slab {n+1:02d} {str(('symmetric' if slab.is_symmetric() else 'non-symmetric') + ', ' + ('stoichiometric' if [float(x)%float(y) for x, y in zip(composition_slab, composition_ref)] == [0]*len(composition_ref) else 'non-stoichiometric')).rjust(32)}"]
    compiled_list.append("----------------------------------------")
    compiled_list.append(f"compos:  {composition.rjust(31)}")
    compiled_list.append(f"dipole:  {str(slab.dipole[2]).rjust(31)}\n")
    compiled_list.append(f"scale :        1.0        1.0  {format(layers/layers_ref, '.1f').rjust(9)}")
    compiled_list.append(f"A/t/v :  {format(slab.surface_area, '.6f').rjust(9)}  {format(thickness, '.6f').rjust(9)}  {format(vacuum, '.6f').rjust(9)}")
    compiled_list.append('\n'.join(str(slab).strip().split('\n')[6:8])+'\n')
    compiled_list.append("----------------------------------------")
    compiled_list.append("termin:    height:   atom configuration:\n")

    for key, values in elements:
        if key == elements[0][0]:
            termination = "UP" if t2 != t1 else "SY"

        elif key == elements[-1][0]:
            termination = "DN" if t2 != t1 else "SY"

        else:
            termination = "..".rjust(3)

        compiled_list.append(f"    {termination.rjust(3)}  {format(key, '.6f').rjust(9)}   {' / '.join(values)}")
        compiled_list.append("")

    compiled_list.append("----------------------------------------")
    compiled_list.append(f"{'UP' if t2 != t1 else 'SY'} ads:  {str(len(frac_ads_sites['ontop'])).rjust(2)} on-top  {str(len(frac_ads_sites['bridge'])).rjust(2)} bridge  {str(len(frac_ads_sites['hollow'])).rjust(2)} hollow\n")

    for pos in ['ontop', 'bridge', 'hollow']:
        for site in frac_ads_sites[pos]:
            compiled_list.append(f" {pos if pos != 'ontop' else 'on-top'}  {format(site[0], '.6f').rjust(9)}  {format(site[1], '.6f').rjust(9)}  {format(site[2], '.6f').rjust(9)}")

    compiled_list.append("")

    if t1 != t2:
        compiled_list.append(f"DN ads:  {str(len(inverse_frac_ads_sites['ontop'])).rjust(2)} on-top  {str(len(inverse_frac_ads_sites['bridge'])).rjust(2)} bridge  {str(len(inverse_frac_ads_sites['hollow'])).rjust(2)} hollow\n")

        for pos in ['ontop', 'bridge', 'hollow']:
            for site in inverse_frac_ads_sites[pos]:
                compiled_list.append(f" {pos if pos != 'ontop' else 'on-top'}  {format(site[0], '.6f').rjust(9)}  {format(site[1], '.6f').rjust(9)}  {format(site[2], '.6f').rjust(9)}")

    compiled_list.append("\n########################################\n")

    if n + 1 == input_output or input_output == 0:
        poscar = Poscar(slab, sort_structure=True)
        poscar.write_file(str(''.join(map(str, input_miller_indices))) + ('SYM' if input_symmetry == True else '') + 'slab' + str(n+1).zfill(2) + ('up' if t2 != t1 else 'sy') + '.vasp')

        fig = plt.figure(); ax = fig.add_subplot(111); plot_slab(slab, ax, adsorption_sites=True); ax.set_xlim(min(slab.lattice.matrix[0][0],slab.lattice.matrix[1][0])-2, max(slab.lattice.matrix[0][0],slab.lattice.matrix[1][0])+2); ax.set_ylim(min(slab.lattice.matrix[0][1],slab.lattice.matrix[1][1])-2, max(slab.lattice.matrix[0][1],slab.lattice.matrix[1][1])+2); plt.savefig(str(''.join(map(str, input_miller_indices))) + ('SYM' if input_symmetry == True else '') + 'slab' + str(n+1).zfill(2) + ('up' if t2 != t1 else 'sy') + '.png')

        if t1 != t2:
            poscar = Poscar(inverse_slab, sort_structure=True)
            poscar.write_file(str(''.join(map(str, input_miller_indices))) + ('SYM' if input_symmetry == True else '') + 'slab' + str(n+1).zfill(2) + 'dn' + '.vasp')

            fig = plt.figure(); ax = fig.add_subplot(111); plot_slab(inverse_slab, ax, adsorption_sites=True); ax.set_xlim(min(slab.lattice.matrix[0][0],slab.lattice.matrix[1][0])-2, max(slab.lattice.matrix[0][0],slab.lattice.matrix[1][0])+2); ax.set_ylim(min(slab.lattice.matrix[0][1],slab.lattice.matrix[1][1])-2, max(slab.lattice.matrix[0][1],slab.lattice.matrix[1][1])+2); plt.savefig(str(''.join(map(str, input_miller_indices))) + ('SYM' if input_symmetry == True else '') + 'slab' + str(n+1).zfill(2) + 'dn' + '.png')

        # convert_poscar_coords('POSCAR.slab' + str(n+1).zfill(2))

    return compiled_list

args = sys.argv[1:] if len(sys.argv) > 1 else []
args_dict = {}

for arg in args:
    name, value = arg.split('=')
    args_dict[name] = value

# usage: python slabs.py input=CONTCAR miller=100 symmetry=False thickness=10 vacuum=15 output=0
input_structure = args_dict['input'] if 'input' in args_dict else "CONTCAR"
input_miller_indices = (int(args_dict['miller'][0]), int(args_dict['miller'][1]), int(args_dict['miller'][2])) if 'miller' in args_dict else (1,0,0)
input_symmetry = eval(args_dict['symmetry']) if 'symmetry' in args_dict else False
input_slab_size = int(args_dict['thickness']) if 'thickness' in args_dict else 10 if input_symmetry == False else 10
input_vacuum_size = int(args_dict['vacuum']) if 'vacuum' in args_dict else 15
input_output = int(args_dict['output']) if 'output' in args_dict else 0

structure = Poscar.from_file(input_structure).structure
oxi_transformation = AutoOxiStateDecorationTransformation()
structure_oxi = oxi_transformation.apply_transformation(structure)

# Use the SlabGenerator class to generate the slab
slabgen = SlabGenerator(structure_oxi, input_miller_indices, min_slab_size=input_slab_size, min_vacuum_size=input_vacuum_size, center_slab=True)
slabs = slabgen.get_slabs(ftol=1e-12, symmetrize=input_symmetry)

slabgen_ref = SlabGenerator(structure_oxi, input_miller_indices, min_slab_size=1, min_vacuum_size=0)
slabs_ref = slabgen_ref.get_slabs(ftol=1e-12)
sites_ref = list(slabs_ref[0].sites)
layers_ref = len([*set([round(site.c,6) for site in sites_ref])])

full_compiled_list = []
ti = 1
for n, slab in enumerate(slabs):

    slab = slab.get_orthogonal_c_slab()

    if slab.is_symmetric() == True:
        tf = ti
    else:
        tf = ti + 1

    # slab_dict[f"Slab {n+1:02d}"] = [("T"+str(t1),"T"+str(t2)),compile_elements(n,t1,t2,slab)]
    full_compiled_list.append(compile_elements(n,ti,tf,slab))

    ti = tf + 1

with open(str(''.join(map(str, input_miller_indices))) + ('SYM' if input_symmetry == True else '') + '.out', 'w', encoding='utf-8') as f:
    print(header, file=f)
    print(f'\ninput:\n', file=f)
    print(f'  bulk structure: {"".join(str(structure.composition.formula).split(" "))}', file=f)
    print(f'  miller indice: {input_miller_indices}', file=f)
    print(f'  min slab thickness: {input_slab_size}', file=f)
    print(f'  min vacuum size: {input_vacuum_size}', file=f)
    print(f'  symmetric: {input_symmetry}', file=f)
    print(f'\nsummary:\n', file=f)
    print(f'  slabs: {len(slabs)}', file=f)
    print(f'  terminations: {tf}', file=f)
    print(f'\n########################################', file=f)

    for compiled_list in full_compiled_list:
        for entry in compiled_list:
            print(entry, file=f)
