from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vb_ege.compat import import_pandas_quietly
from vb_ege.plotting import plot_symmetric_hard_scaling_grid


pd = import_pandas_quietly()

scaling = pd.read_csv(ROOT / "results" / "summary" / "fixed_confidence_scaling_summary.csv")
confidence = pd.read_csv(ROOT / "results" / "summary" / "confidence_scaling_quantile_summary.csv")
output = ROOT / "results" / "figures" / "fixed_confidence_scaling" / "symmetric_hard_scaling_2x2.pdf"

plot_symmetric_hard_scaling_grid(scaling, confidence, output)
print(output)
