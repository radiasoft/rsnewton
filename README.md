# rsnewton
Simulation tools to support design and control studies for high intensity superconducting linacs

# Installation
`pip install -e .`

# Lattice trimming utility 
Creates a new opal sim with the initial bunch distribution computed at the start element

`rsnewton lattice trim-lattice template.i CAV060 D234 --output-dir CAV060_to_end --opal-bin opal --mpi-ranks 16`

- template.i — the full lattice file
- CAV060 — start element (trim precomputes the real tracked beam up to here)
- D234 — end element (last placed element in BL1's LINE=(...))
- --output-dir — optional, defaults to CAV060_to_end next to the input if omitted
- --opal-bin/--mpi-ranks — optional; use --mpi-ranks to run the precompute step under mpirun -np 16 instead of serial opal
