---
icon: fontawesome/brands/python
---

# Pointwise operations

[`XTensor`][fiery.xtensor.XTensor]'s elementwise arithmetic, comparison,
logical, and trigonometric operations — each behaves exactly like the
matching `torch.*` op, with this tensor's names (and coordinates, where
applicable) propagated onto the result.

!!! note "Not one of the issue's originally proposed sections"
    [Issue #155](https://github.com/bagofseeds/fiery-xtensor/issues/155)'s
    candidate list didn't call out `_pointwise.py` on its own; its ~40 methods
    were folded into "Core" by omission. They're numerous and self-contained
    enough to warrant their own page rather than padding out another section.

::: fiery.xtensor.XTensor
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - acos
        - acosh
        - add
        - asin
        - asinh
        - atan
        - atan2
        - atanh
        - cos
        - cosh
        - div
        - eq
        - erf
        - erfc
        - exp
        - expm1
        - floor_divide
        - ge
        - gt
        - hypot
        - le
        - log
        - log10
        - log1p
        - log2
        - logical_and
        - logical_or
        - logical_xor
        - lt
        - maximum
        - minimum
        - mul
        - ne
        - pow
        - remainder
        - sigmoid
        - sin
        - sinh
        - sub
        - tan
        - tanh
