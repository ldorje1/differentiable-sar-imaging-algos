# differentiable-sar-imaging-algos

This repository contains MATLAB implementations of classical SAR imaging algorithms reformulated as differentiable operators for gradient-based optimization. The project focuses on making standard image formation methods such as MFA, RMA, BPA, LIA, and CSA differentiable so gradients can be propagated through the reconstruction pipeline and used with loss-driven optimization. In this work, the imaging process is treated as part of an end-to-end optimization framework rather than only a fixed reconstruction step.


***

<p align="center">
  <img src="figures/sar_image_diff_algo.png" alt="SAR image differentiation algorithm" width="600">
</p>
