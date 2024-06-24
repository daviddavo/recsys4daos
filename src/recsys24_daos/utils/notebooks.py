"""Utils for use in Jupyter Notebooks
"""
import sys
import io

from matplotlib import pyplot as plt

def print_versions(*args):
    print(sys.version)
    for m in ['recommenders', *args]:
        try:
            print(f'{m}:', sys.modules[m].__version__)
        except KeyError:
            print(f'module {m} not loaded')

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
