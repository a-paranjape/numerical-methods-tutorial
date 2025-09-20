# Numerical methods tutorial
## Pedagodical introduction to basic numerical methods

* Finite-differencing schema
  - ODEs : Verlet integration, Forward/Backward Euler, Crank-Nicholson, Euler-Cromer, Stoermer-Verlet
  - PDEs : *under construction*

## Code organization
* `code/ode.py`
  - `ODEInt`: schema for generic 2nd-order ODEs of the type $\ddot u(t) = f(u,t) + g(v)$ with $v = \dot u$, including Verlet, Forward Euler, Euler-Cromer and Stoermer-Verlet.

* `code/examples.py`
  - Support for various choices of $f(u,t)$ and $g(v)$ for use in `ODEInt`, including simple harmonic oscillator, ...
    
* `code/tests.py`
  - `Test_ODEInt`: error and error convergence rate calculations for class ODEInt

## Example notebooks
* $\texttt{examples/ODE\\_Examples.ipynb}$: Examples showing basic usage and convergence tests on SHO.

## Requirements
* Python3.8 or higher with NumPy, Matplotlib and Jupyter (for example usage).

## Installation
* Clone into this repo and start using.

## Contact
Aseem Paranjape: aseem_at_iucaa_dot_in
