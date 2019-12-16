# Data 

We have provided two primary datasets (ZINC and CASF), for which we have pre-computed fragmentations and structural information. These datasets match the dataset utilised in our paper, [Deep Generative Models for 3D Compound Design](https://www.biorxiv.org/content/10.1101/830497v1).

We have also provided several scripts to allow you to use your own dataset.

# To preprocess your own dataset

## Option 1: Pre-computed fragmentations

If you have prepared your own fragmentations (two unlinked substructures), run `calculate_distance_angle.py`. 

You will need to supply a data file containing a list of fragments and molecules, and an SD file containing a conformation of each molecule.

```
python calculate_distance_angle.py --data_path PATH_TO_FILE --sdf_path PATH_TO_FILE --output_path PATH_TO_FILE --verbose
```

The format of the data file is: 

```
Fragments (SMILES) Full molecule (SMILES)
```

For example:

```
COc1ccccc1[*:2].Fc1cccc([*:1])c1 COc1ccccc1CCC(=O)c1cccc(F)c1
```

Now run `prepare_data.py` with the output file as the first argument (see below for details).

## Option 2: SD file of molecules

If you want to simply provide an SD file containing a set of molecules, run `prepare_data_from_sdf.py`.

```
python prepare_data_from_sdf --sdf_path PATH_TO_DATA --output_path PATH_TO_FILE --verbose
```

This will compute fragmentations and structural information, as per the criteria described in our paper, [Deep Generative Models for 3D Compound Design](https://www.biorxiv.org/content/10.1101/830497v1). If you do not want to filter the fragmentations using the 2D chemical property filters described in our paper, add the flag `--no_filters` to the above command.

For example, the following command reproduces the CASF dataset entries in `data_casf_final.txt`.

```
python calculate_distance_angle.py --sdf_path ../analysis/casf_structures.sdf --output_path data_casf_duplicate.txt --verbose
```

Now run `prepare_data.py` with the output file as the first argument (see below for details).

# To use a provided dataset

To process the provided datasets (ZINC and CASF), run `prepare_data.py`. This allows you to train, validate and generate molecules using `DeLinker.py`. Generated molecules will have a linker with at most the same number of atoms as the reference molecule provided.

```
python prepare_data.py
```

If you want to process your own dataset (having followed the above preprocessing steps), run `prepare_data.py` with the following arguments:

```
python prepare_data.py --data_path PATH_TO_DATA --dataset_name NAME_OF_DATASET
```

The format taken by `prepare_data.py` is: 

```
Full molecule (SMILES) Linker (SMILES) Fragments (SMILES) Distance (Angstrom) Angle (Radians)
```

For example:

```
COc1ccccc1CCC(=O)c1cccc(F)c1 O=C(CC[*:2])[*:1] COc1ccccc1[*:2].Fc1cccc([*:1])c1 4.69 2.00
```

If you want to use `DeLinker_test.py` (which generates linkers with a specified number of atoms), run `prepare_data` with the following arguments:

```
python prepare_data.py --data_path PATH_TO_DATA --dataset_name NAME_OF_DATASET --test_mode
```

`prepare_data.py` takes two possible input formats, listed below.

```
Fragments (SMILES) Distance (Angstrom) Angle (Radians)
Full molecule (SMILES) Linker (SMILES) Fragments (SMILES) Distance (Angstrom) Angle (Radians)
```


# Contact (Questions/Bugs/Requests)

Please submit a Github issue or contact Fergus Imrie [imrie@stats.ox.ac.uk](mailto:imrie@stats.ox.ac.uk).

