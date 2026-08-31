# Add a square-root approximation law

## Summary

Extend `floating_real_like` with the missing law relating a floating-point
`sqrt` to `FStar.Math.Sqrt.sqrt` for nonnegative real models.  Also add the
usual explicitly-triggered wrapper, matching the existing `exp`, `div`, and
`log` approximation laws.

This lets clients state nonlinear specifications directly over real-valued
inputs instead of introducing unconstrained or implementation-shaped
floating-point square-root witnesses.

## Scope

The patch adds no executable code and no new real square-root definition.
`FStar.Math.Sqrt.rnonneg` provides the domain and `FStar.Math.Sqrt.sqrt` is the
real model.  Implementations of `floating_real_like` acquire one additional
semantic proof obligation.

## Validation

Applied `kuiper-sqrt-approx.patch` to nightly `2026-08-31` and checked the
patched `Kuiper.Approximates.Base` with the packaged F* configuration.  All
verification conditions discharged.
