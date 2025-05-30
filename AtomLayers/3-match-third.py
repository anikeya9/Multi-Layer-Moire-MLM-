from tools.tool import load_layer_basis
from tools.match import run
import polars as pl
from fire import Fire
import numpy as np
from tqdm import tqdm
from pathlib import Path
from icecream import ic


def main(firstlayer, priormatches, tolerance, outdir):
    A = load_layer_basis(firstlayer)
    pmatchdf = pl.read_csv(priormatches)

    nrows = len(pmatchdf)
    for i in tqdm(range(nrows)):
        row = pmatchdf[i]
        # ic(row)

        v1 = row["v1x"][0], row["v1y"][0]
        v2 = row["v2x"][0], row["v2y"][0]
        G = np.array([v1, v2]).T

        outdf = run(A, G, tolerance=tolerance)
        outfile = Path(outdir) / f"{i}.csv"
        outdf.write_csv(outfile)


if __name__ == "__main__":
    Fire(main)
