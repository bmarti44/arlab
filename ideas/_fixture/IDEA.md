# _fixture — CPU test pack for the arlab runner (not a research idea)

A tiny numpy MLP classifies a synthetic 10-class, 32-dimensional dataset. The baseline learning rate is
deliberately too low, so a known better setting exists (raise `lr`). Seeds change init and batch order.
The default is a seed pack (no items). Used by `make accept-M1` with a scripted backend; tests make variants
(items on, deterministic `confirm: []`, different MES).
