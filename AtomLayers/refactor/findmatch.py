import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from match import run_and_filter, run, filtermatches
    from tool import angle_in_degrees

    from pprint import pprint
    return np, pprint, run, run_and_filter


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
    results = run_and_filter(a1,a2,g1,g2,degmin=20,degmax=30,degstep=0.01,nmin=-20,nmax=20,dtol=3e-4, goodangles=[60.0])
    return (results,)


@app.cell
def _(results):
    results[0]["results"][-40:]
    return


@app.cell
def _(np):
    _a = np.array([2.467291, 8.546948])
    _b = np.array([-6.168228, 6.410211])

    _a /= np.linalg.norm(_a)
    _b /= np.linalg.norm(_b)

    np.dot(_a,_b), np.cos(np.deg2rad(120))
    return


@app.cell
def _(pprint, results):
    for _r in results:
        pprint(_r["results"])
    return


@app.cell
def _():
    # results[0]["results"][10]
    return


@app.cell
def _(a1, a2, g1, g2, run):
    r = run(a1,a2,g1,g2,degmin=0,degmax=30,degstep=0.01,nmin=-20,nmax=20,dtol=5e-4)

    return (r,)


@app.cell
def _(r):
    for _a in r["angle"].unique():
        print(_a)
    return


app._unparsable_cell(
    r"""
    for _r in filtermatches(a1,a2,r,goodangle=120):

    """,
    name="_"
)


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
