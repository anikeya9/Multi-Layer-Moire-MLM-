import polars as pl
import numpy as np
from fire import Fire

from tools.tool import load_layer_basis,angle_in_degrees


def main(
    matchcsv,
    firstlayer,
    #secondlayer,
    outfile,
    good_basis_angles=[90, 60],
):
    matches_with_good_basis = []

    A = load_layer_basis(firstlayer)
    #G = load_layer_basis(secondlayer)

    df = pl.read_csv(matchcsv)
    angles = df["angle"].unique()
    for ang in angles:
        rows = df.filter(df["angle"] == ang).sort("norm_m1m2")
        top4 = rows[:4]
        norms = top4["m1"] ** 2 + top4["m2"] ** 2
        assert (
            norms.unique().shape[0] == 1
        )  # make sure they all are just +- of each other
        matrix = top4["n1", "n2"].to_numpy()
        bases = []
        for i in range(4):
            for j in range(i):
                v1 = A @ matrix[i, :]
                v2 = A @ matrix[j, :]
                bases.append(
                    {
                        "angle": ang,
                        "v1x": v1[0],
                        "v1y": v1[1],
                        "v2x": v2[0],
                        "v2y": v2[1],
                        "angle_v1v2": angle_in_degrees(v1, v2),
                    }
                )
        # select best angle and also see if the 1st vector can be in the first quadrant.
        bases = sorted(
            bases,
            key=lambda b: np.abs(b["angle_v1v2"] - good_basis_angles[0])
            - np.sign(
                b["v1x"]
            )  # heuristic to make the 1st basis vector v1 in first quadrant
            - np.sign(b["v1y"]),
        )

        best = pl.from_dicts(
            bases,
            schema={
                "angle": pl.Float64,
                "v1x": pl.Float64,
                "v1y": pl.Float64,
                "v2x": pl.Float64,
                "v2y": pl.Float64,
                "angle_v1v2": pl.Float64,
            },
        )[0]  # pick only the best match

        matches_with_good_basis.append(best)

    out = pl.concat(matches_with_good_basis)

    out.write_csv(outfile)


if __name__ == "__main__":
    Fire(main)
