"""Group F — black-box and transfer attacks (plan §2).

All slots open. These attacks may call ``ctx.model.predict`` (query-based) but
must set ``needs_gradients = False``, so they also run against adapters that
expose no gradients.

Two entries matter for trustworthy numbers, per plan §11:

* a random-noise baseline at the same L-inf budget — if a gradient attack is not
  clearly stronger than random noise, the gradient attack is broken;
* Square Attack (Andriushchenko et al., ECCV 2020) plus a transfer matrix, which
  catch gradient masking that white-box numbers alone would hide.
"""
