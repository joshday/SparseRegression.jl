# Algorithms

The first argument of an `Algorithm`'s constructor is an `SModel`.  `Algorithm` subtypes hold storage buffers and this ensures the buffers are the correct size.

```@docs
ProxGrad
Fista
GradientDescent
Sweep
LinRegCholesky
```
