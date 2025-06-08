import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from match import run_and_filter, run, filtermatches
    from tool import angle_in_degrees
    return filtermatches, np, run, run_and_filter


@app.cell
def _():
    # def match(a1,a2,g1,g2,degmin,degmax,degstep,nmin,nmax,dtol,goodangles=[60.0,90.0,120.0]):
    # return deg,n1,n2,n1p,n1p,delvec,delgoodangle,v1,v2
    return


@app.cell
def _(np):
    np.radians(180)
    return


@app.cell
def _():
    a1 = [1.2336456308, -2.1367369111]
    a2 = [1.2336456308, 2.1367369111]
    g1 = a1
    g2 = a2
    return a1, a2, g1, g2


@app.cell
def _(a1, a2, g1, g2, run_and_filter):
    results = run_and_filter(a1,a2,g1,g2,degmin=0,degmax=30,degstep=0.01,nmin=-20,nmax=20,dtol=1e-4, goodangles=[120.0])
    return (results,)


@app.cell
def _(results):
    results
    return


@app.cell
def _():
    # results[0]["results"][10]
    return


@app.cell
def _(a1, a2, g1, g2, run):
    r = run(a1,a2,g1,g2,degmin=0,degmax=30,degstep=0.01,nmin=-20,nmax=20,dtol=1e-4)

    return (r,)


@app.cell
def _(np, r):
    v1 = np.array(r[0]["matchpoint"].to_numpy()[0], dtype=float)
    v1 /= np.linalg.norm(v1)
    return (v1,)


@app.cell
def _(np, r):
    v2 = np.array(r[2]["matchpoint"].to_numpy()[0], dtype=float)
    v2 /= np.linalg.norm(v2)
    np.linalg.norm(v2)
    return (v2,)


@app.cell
def _(np, v1, v2):
    np.rad2deg(np.arccos(np.dot(v1,v2)))
    return


@app.cell
def _(r):
    r
    return


@app.cell
def _(a1, a2, filtermatches, r):
    filtermatches(a1,a2,r,goodangle=120)
    return


@app.cell
def _(a1, a2, np, r):
    f = r.filter(r["angle"] == 9.43)
    V = np.array(f["matchpoint"].to_list())
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    # np.linalg.norm(V, axis=1, keepdims=True)

    N = f["n1","n2"].to_numpy()
    V1 = N @ np.array([a1, a2])
    V1 /= np.linalg.norm(V1, axis=1, keepdims=True)
    return N, V, V1


@app.cell
def _(V1):
    V1[:,0]**2 + V1[:,1]**2
    return


@app.cell
def _(V, np):
    dot = V@ V.T
    cos = np.cos(np.deg2rad(120))

    ix,iy = np.where(np.isclose(dot, cos))
    ix,iy
    return cos, dot


@app.cell
def _(cos, dot):
    dot - cos
    return


@app.cell
def _(N):
    N
    return


@app.cell
def _(N):
    N.T 
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
