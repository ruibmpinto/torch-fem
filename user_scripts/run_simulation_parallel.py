"""Parallel FEM simulation using PyTorch DataLoader."""
#
#                                                                       Modules
# =============================================================================
import torch
from torch.utils.data import Dataset, DataLoader
from run_simulation import Simulation
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rui_pinto@brown.edu)'
__credits__ = ['Rui B. M. Pinto', ]
__status__ = 'Development'
# =============================================================================
torch.set_default_dtype(torch.float64)
# =============================================================================
class SimulationDataset(Dataset):
    """Dataset for parallel FEM simulations.

    Each item represents a complete simulation configuration. Worker
    processes execute simulations independently and return results.
    """

    def __init__(self, configs):
        """Initialize dataset with simulation configurations.

        Args:
            configs (list): List of dicts, each containing parameters
                for Simulation.__init__
        """
        self.configs = configs

    def __len__(self):
        return len(self.configs)

    def __getitem__(self, idx):
        """Run simulation for given configuration index.

        Exceptions raised by Simulation.run() are caught and
        returned as a failure dict rather than propagated.
        This prevents a single failure from aborting the
        DataLoader pipeline and losing all queued
        simulations downstream of the failing index.

        Args:
            idx (int): Configuration index

        Returns:
            dict: On success, contains patch_idx,
                simulation_data, config, and status='ok'. On
                failure, contains patch_idx, config,
                status='failed', error message, and
                traceback.
        """
        import traceback
        config = self.configs[idx]
        try:
            sim = Simulation(**config)
            sim.run()
            return {
                'patch_idx': config['patch_idx'],
                'simulation_data': sim.simulation_data,
                'config': config,
                'status': 'ok',
            }
        except Exception as excp:
            return {
                'patch_idx': config['patch_idx'],
                'config': config,
                'status': 'failed',
                'error': (
                    f'{type(excp).__name__}: {excp}'),
                'traceback': traceback.format_exc(),
            }
# =============================================================================
def _collate_single(batch):
    """Unwrap single-item batch for DataLoader collation.
    
    With batch_size=1:

    - Worker calls __getitem__(idx) once → returns single simulation result
    - collate_fn receives [result] (list with one element)
    - _collate_single unpacks to return result directly
    
    With  batch_size=4:
    
    - Worker calls __getitem__ 4 times → 4 independent simulations run
        sequentially in that worker
    - collate_fn receives [result1, result2, result3, result4]
    """
    return batch[0]
# =============================================================================
def run_parallel_simulations(
    configs,
    num_workers=4,
    verbose=True,
    failure_log=None,
):
    """Execute multiple simulations in parallel using DataLoader.

    Args:
        configs (list): List of simulation configuration dicts
        num_workers (int): Number of parallel worker processes
        verbose (bool): Print progress information
        failure_log (str or None): If provided, path to a
            text file where one line per failed patch
            (patch_idx<TAB>error) is written after the run.

    Returns:
        list: Simulation results from all workers. Each
            result carries status='ok' or status='failed'.
    """
    dataset = SimulationDataset(configs)

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=_collate_single,
        persistent_workers=False
    )

    results = []
    failures = []
    for i, result in enumerate(loader):
        if result.get('status') == 'failed':
            failures.append(result)
            if verbose:
                print(
                    f'FAIL idx={result["patch_idx"]}: '
                    f'{result["error"]}')
        if verbose and i % 100 == 0:
            print(
                f'Completed {i}/{len(configs)} simulations')
        results.append(result)

    if verbose:
        n_ok = len(results) - len(failures)
        print(
            f'Summary: {n_ok}/{len(results)} ok, '
            f'{len(failures)} failed')

    if failure_log and failures:
        with open(failure_log, 'w') as f_log:
            for fail in failures:
                f_log.write(
                    f'{fail["patch_idx"]}\t'
                    f'{fail["error"]}\n')

    return results
# =============================================================================
if __name__ == '__main__':
    filepath = '/Volumes/Expansion/material_patches_data/'

    configs = []
    for idx in range(1648, 1670):
        configs.append({
            'element_type': 'quad4',
            'material_behavior': 'elastoplastic_nlh',
            'num_increments': 100,
            'patch_idx': idx,
            'filepath': filepath,
            'mesh_nx': 8,
            'mesh_ny': 8,
            'mesh_nz': 1,
            'is_save': True,
            'is_red_int': False,
            'is_compute_stiffness': False,
            'is_save_avg_epbar': False,
            'is_save_nodal_epbar': False,
            'is_adaptive_timestepping': True,
            'adaptive_max_subdiv': 8,
        })

    results = run_parallel_simulations(
        configs,
        num_workers=8,
        verbose=True,
        failure_log='failures.log',
    )

    print(f'Completed!')
