# Parameter-Estimation-BTP
Script files to run Hard et. al. 2018

MATLAB <br>
**Hard_et_al_test**: Online learning pipeline, choose data input file and set hyperparameters before running (script file) <br>
**Hard_et_al_test_01**: Mini-batch learning pipeline, choose data input file and set hyperparameters before running (script file) <br>
**Had_et_al_test_projection_quad**: Quadratic projection into $$B_{\alpha}$$, MATLAB function file (not a script file), call function from a separate script or live script <br>
**Had_et_al_test_simulate_data**: Simulates the system response to given $G(z)$ for using $(a,B,C,D)$ <br>
**Had_et_al_test_impulse_response**: For a given $G(z)$ outputs the l2 norm of the impulse response, to validate if $|r|$ is greater than the offset $D$<br>

Python<br>
**JMLR 001**: Plots $p(z)$ for visually verifying acquiescense of a function, input $p(z)$ using roots of polynomial or it's coefficients

Developed by **[Akshaj Aithal](https://github.com/Akshaj0712)**.

This project is based on:
- Research literature from Hardt, M., Ma, T., & Recht, B. (2018). [Gradient descent learns linear dynamical systems.](https://jmlr.org/papers/volume19/16-465/16-465.pdf) Journal of Machine Learning Research, 19(29) 
