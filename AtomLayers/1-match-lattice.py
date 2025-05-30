from tools.match import run
from fire import Fire
from tools.tool import load_layer_basis


def main(firstlayer, secondlayer, outfile, tolerance):
    # bottom layer (Alpha coordinate system) with respect to (wrt) the standard basis (Iota)
    # a1, a2 = [3.99, 0], [0, 3.99]
    # create the change of basis matrix A for bottom layer (change from Alpha to Iota)
    # A = np.vstack(np.array([a1, a2])).T

    # basis top layer (Gamma coordinate system) with respect to standard basis (Iota)
    # g1, g2 = [3.99, 0], [0, 3.99]
    # create the change of basis matrix G for top layer (change from Gamma to Iota)
    # G = np.vstack(np.array([g1, g2])).T

    A, G = list(map(load_layer_basis, [firstlayer, secondlayer]))
    # print("A\n", A)
    # print("G\n", G)
    df = run(A, G, tolerance=tolerance)
    df.write_csv(outfile)
    print(f"Stored results in {outfile}")


if __name__ == "__main__":
    Fire(main)
