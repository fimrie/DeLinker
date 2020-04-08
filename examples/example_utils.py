from rdkit import Chem

def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    tmp_mol = Chem.Mol(mol)
    for idx in range(atoms):
        tmp_mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(tmp_mol.GetAtomWithIdx(idx).GetIdx()))
    return tmp_mol

def unique_mols(sequence):
    seen = set()
    return [x for x in sequence if not (tuple(x) in seen or seen.add(tuple(x)))]
