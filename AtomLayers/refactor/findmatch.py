import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from match import run_and_filter
    from tool import angle_in_degrees
    return np, run_and_filter


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
    a1 = [3.9903791,0]
    a2 = [0,3.9903791]
    g1 = a1
    g2 = a2
    return a1, a2, g1, g2


@app.cell
def _(a1, a2, g1, g2, run_and_filter):
    results = run_and_filter(a1,a2,g1,g2,degmin=0,degmax=30,degstep=0.01,nmin=-20,nmax=20,dtol=1e-4, goodangles=[90.0,60.0])

    return (results,)


@app.cell
def _(results):
    results[0]["results"][10]
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
