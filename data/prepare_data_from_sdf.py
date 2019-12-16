#!/usr/bin/env/python
"""
Usage:
    prepare_data_from_sdf.py [options]

Options:
    -h --help                Show this screen
    --sdf_path FILE          Path to SD file containing conformers of reference molecules
    --output_path FILE       Path to output file
    --no_filters             Do not check 2D chemical properties of fragmentations
    --verbose                Print progress and updates to terminal
"""

from rdkit import Chem
from rdkit.Chem import AllChem

import frag_utils

from docopt import docopt

if __name__ == "__main__":
    # Parse args
    args = docopt(__doc__)
    sdf_path = args.get('--sdf_path')
    output_path = args.get('--output_path')
    no_filters = args.get('--no_filters')
    verbose = args.get('--verbose')
    
    # Load data
    conformers = Chem.SDMolSupplier(sdf_path)
    smiles = []
    errors = 0
    for num, sdf in enumerate(conformers):
        try:
            smiles.append(Chem.MolToSmiles(sdf))
        except:
            errors +=1
    if verbose:
        print("Original num entries: \t%d" % len(conformers))
        print("Number parsed by RDKit: %d" % len(smiles))
        print("Number of errors: \t%d" % errors)

    # Filter SMILES for permitted atom types
    smiles_filt = []
    errors = 0
    for i, smi in enumerate(smiles):
        if frag_utils.check_smi_atom_types(smi):
            smiles_filt.append(smi)
        else:
            errors +=1
    
    if i % 1000 == 0 and verbose:
        print("\rProcessed smiles: %d" % i, end='')
        
    if verbose:
        print("Original num entries: \t\t\t%d" % len(smiles))
        print("Number with permitted atom types: \t%d" % len(smiles_filt))
        print("Number of errors: \t\t\t%d" % errors)

    # Fragment dataset
    fragmentations = frag_utils.fragment_dataset(smiles_filt, linker_min=3, fragment_min=5, min_path_length=2, linker_leq_frags=True, verbose=True)

    if verbose:
        print("Processed smiles: \t%d" % len(smiles_filt))
        print("Num fragmentations: \t%d" % len(fragmentations))

    # Filter fragmentions based on 2D properties
    if no_filters:
        fragmentations_filt = fragmentations
    else:
        fragmentations_filt = frag_utils.check_2d_filters_dataset(fragmentations, n_cores=3)

    print("Number fragmentations: \t\t%d" % len(fragmentations))
    print("Number passed 2D filters: \t%d" % len(fragmentations_filt))

    # Calculate structural information
    fragmentations_new, distances, angles, fails = frag_utils.compute_distance_and_angle_dataset(fragmentations_filt, sdf_path, dataset="CASF", verbose=True)

    if verbose:
        print("Number of successful fragmentations: \t\t%d" % len(distances))
        print("Number failed fragmentations: \t\t\t%d" % fails[0]) 
        print("Number of SDF errors: \t\t\t%d" % fails[1])
        print("Number of fragmentations without conformers: \t%d" % (len(fragmentations_filt)-len(fragmentations_new)))


    # Write data to file
    # Format: full_mol (SMILES), linker (SMILES), fragments (SMILES), distance (Angstrom), angle (Radians)
    with open(output_path, 'w') as f:
        for fragmentation, dist, ang in zip(fragmentations_new, distances, angles):
            f.write("%s %s %s %s %s\n" % (fragmentation[0], fragmentation[1], fragmentation[2], dist, ang))
            
    print("Done")
