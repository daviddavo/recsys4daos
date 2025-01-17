"""Utils for use in Jupyter Notebooks
"""
from typing import Any

import sys
import io
from pathlib import Path
from dataclasses import dataclass, asdict, field
import warnings

import nbformat
from matplotlib import pyplot as plt

@dataclass(frozen=True)
class DaoToRun:
    # Name of the organization (see ./data/input)
    org_name: str
    # Frequency of the splits
    splits_freq: str = '7d'
    # Wether to normalize the folds (start at 00:00)
    splits_normalize: bool = True
    # Number of folds to use
    last_folds: int = 10
    # Date of the last fold to use
    last_fold_date: str = None
    # Run until this notebook number (inclusive)
    run_until_nb: int = 100
    # Note or comment about why the dao is removed
    comment: str = ""
    # Extra notebook hyperparameters
    extra_hparams: dict[str, Any] = field(default_factory=lambda:{})


def print_versions(*args):
    print(sys.version)
    for m in ['recommenders', *args]:
        try:
            print(f'{m}:', sys.modules[m].__version__)
        except KeyError:
            print(f'module {m} not loaded')

def is_increasing(l):
    return (not l) or all(x<y for x,y in zip(l, l[1:]))

def isCompleted(fname):
    nb = nbformat.read(fname, as_version=4)
    execution_counts = [c['execution_count'] for c in nb['cells'] if c['cell_type'] == 'code' and c['source'].strip()]
    
    return all((x is not None for x in execution_counts)) and is_increasing(execution_counts)

def getOldExecID(fname):
    nb = nbformat.read(fname, as_version=4)
    return nb.metadata['papermill']['parameters'].get('EXECUTION_ID')

def run_dao_notebook(fname, output_path, EXECUTION_ID, *, EXTRA_HPARAMS={}, **kwargs):
    import papermill as pm
    
    assert fname.exists(), f"No existe el fichero {fname}"
    
    outpath = Path(output_path) / kwargs['ORG_NAME']
    outpath.mkdir(parents=True, exist_ok=True)
    outfile = outpath/fname.name

    params = pm.inspect_notebook(fname)
    for p in kwargs.keys():
        assert p in params, f'{p} is not in notebook {fname} params'
    for p in EXTRA_HPARAMS:
        if p not in EXTRA_HPARAMS:
            warnings.warn(f'extra hparam {p} is not in notebook {fname} params')

    if outfile.exists() and EXECUTION_ID:
        oldExec = getOldExecID(outfile)
        
        if oldExec == EXECUTION_ID and isCompleted(outfile):
            print(f"Skipping {outfile} with EXECUTION_ID {EXECUTION_ID}")
            return
        elif oldExec != EXECUTION_ID:
            print(f"Different exec, re-running ({oldExec} != {EXECUTION_ID})")
        else:
            print(f"Was not complete, re-running")
    
    pm.execute_notebook(
        fname,
        outfile,
        progress_bar={
            'leave': False,
            'desc': str(outfile),
        },
        autosave_cell_every=30,
        parameters=dict(
            EXECUTION_ID=EXECUTION_ID,
            **EXTRA_HPARAMS,
            **kwargs,
        ),
    )
    # Make readonly
    print("Finished running", str(outfile))

def download_matplotlib(fig: plt.Figure, filename):
    import solara
    format = filename.split('.')[-1]

    def get_data():
        buf = io.BytesIO()
        fig.savefig(buf, format=format, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        return buf

    return solara.FileDownload(
        data=get_data,
        filename=filename,
    )


def download_plotly(fig, filename):
    import solara
    format = filename.split('.')[-1]

    return solara.FileDownload(
        data=lambda: fig.to_image(format=format),
        filename=filename,
    )
