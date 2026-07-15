# Authorized validation input

`aapl_spy_qqq_through_2023.csv` was created only after the frozen selector and
its passing through-2018 checkpoint had been committed and pushed at
`82312e2e231c4b9be856789507f01354521f975b`.

The source is the already tracked, physically through-2023 file:

`e/aapl_sector_breadth_residual_edge_v1/sector-breadth-development-v1/development_prices_through_2023.csv`

Source SHA-256:
`3b5e02acaa69a56fa47a0fd34275472d226b62c0f61741c13d3680b239b82535`.

The packaging transform removed only the redundant `aapl_adj_open` column so
the file has the experiment's exact six-column physical-snapshot schema. It
did not filter, revise, inspect, or select rows based on model performance.
The adjusted open is deterministically reconstructed by the sealed loader from
raw AAPL open, raw AAPL close, and adjusted AAPL close.

Target SHA-256:
`c5189db9796f25ae69d14814b22ac4a852449aef8b86d3615a289b9fcb8029e9`.

The sealed loader independently verifies 6,244 sessions from 1999-03-10
through 2023-12-29, no later physical row, and canonical bounded-result hash
`sha256:3b5e02acaa69a56fa47a0fd34275472d226b62c0f61741c13d3680b239b82535`.
