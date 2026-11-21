# Sokoban solver — make.fun bounty `sokoban-solver-01-run-20261121`

A push-optimal Sokoban solver. The generator builds each level as a 1-wide
hallway with 1–4 rooms off it (one door each, boxes never leave their room),
so the level optimum is the sum of the rooms' optima. This entry decomposes
each level into its rooms and solves every room push-optimally with A* over
pushes — Hungarian-assignment heuristic on BFS distances, dead-square and
2×2 freeze pruning, player-region normalisation, a transposition table, and
fixed expansion budgets (deterministic: no clock, no randomness). The rooms'
move strings are joined with hallway walks and the whole answer is verified
by replay before it is written.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<dir with i001..i005/level.txt> MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22+ (type stripping, erasable TypeScript only, node builtins only).
Writes `out/<id>/solution.txt` per instance. Whole practice set solves in
~0.5 s, all 5 instances at their push optima (P*).
