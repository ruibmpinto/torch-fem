"""Patch model_init_file.pkl to set is_rigid_body_removal=True.

Iterates over all elastic surrogate model directories and ensures
the is_rigid_body_removal flag is present and set to True.

Notes
-----
Run with: mamba run -n env_graphorge python patch_rbm_flag.py
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import sys
import pickle
import pathlib

# Add graphorge to sys.path for pkl deserialization
graphorge_path = str(
    pathlib.Path(__file__).parents[3]
    / 'graphorge_material_patches' / 'src')
if graphorge_path not in sys.path:
    sys.path.insert(0, graphorge_path)
# Some pkl files reference 'utilities' directly
graphorge_inner = str(
    pathlib.Path(__file__).parents[3]
    / 'graphorge_material_patches' / 'src'
    / 'graphorge')
if graphorge_inner not in sys.path:
    sys.path.insert(0, graphorge_inner)


#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'R. Barreira'
__status__ = 'Prototype'


# =============================================================================
#
# =============================================================================
SURROGATES_ROOT = str(
    pathlib.Path(__file__).parents[1]
    / 'matpatch_surrogates' / 'elastic')


def patch_all_pkl_files():
    """Set is_rigid_body_removal=True in all model pkl files."""

    patched = []
    skipped = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for root, dirs, files in os.walk(SURROGATES_ROOT):
        if 'model_init_file.pkl' not in files:
            continue
        pkl_path = os.path.join(
            root, 'model_init_file.pkl')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        model_init_args = data['model_init_args']
        current = model_init_args.get(
            'is_rigid_body_removal', None)
        rel_path = os.path.relpath(
            pkl_path, SURROGATES_ROOT)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if current is True:
            skipped.append((rel_path, 'already True'))
            continue
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        model_init_args[
            'is_rigid_body_removal'] = True
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        patched.append(
            (rel_path, f'was {current}'))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('\n--- Patched ---')
    for path, note in sorted(patched):
        print(f'  {path}: {note}')
    print(f'\n--- Skipped (already True) ---')
    for path, note in sorted(skipped):
        print(f'  {path}: {note}')
    print(f'\nTotal: {len(patched)} patched, '
          f'{len(skipped)} skipped')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Verify all files
    print('\n--- Verification ---')
    for root, dirs, files in os.walk(SURROGATES_ROOT):
        if 'model_init_file.pkl' not in files:
            continue
        pkl_path = os.path.join(
            root, 'model_init_file.pkl')
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        flag = data['model_init_args'].get(
            'is_rigid_body_removal', 'MISSING')
        rel = os.path.relpath(
            pkl_path, SURROGATES_ROOT)
        status = 'OK' if flag is True else 'FAIL'
        print(f'  [{status}] {rel}: {flag}')


# =============================================================================
if __name__ == '__main__':
    patch_all_pkl_files()
