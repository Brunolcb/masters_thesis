from pathlib import Path

from src.utils import plot_segmentation_performance_report


if __name__ == '__main__':
    for results_fpath in Path('data/pred/').glob('*.npz'):
        fig = plot_segmentation_performance_report(results_fpath)
        fig.savefig('models/'+results_fpath.name.replace('npz','png'), dpi=400)
