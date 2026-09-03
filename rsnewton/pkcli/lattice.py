"""Split an OPAL lattice into independently-runnable sub-lattices.

Studies that scan something downstream of a given element (an RF failure, a
misalignment, a field-map swap, ...) don't need to re-simulate the
unperturbed upstream section of the beamline on every iteration. This module
runs a lattice once, truncated at a "start" element, converts the resulting
beam state into an OPAL FROMFILE distribution, and writes a second,
self-contained lattice file spanning "start" through "end" that uses it.

:copyright: Copyright (c) 2026 RadiaSoft LLC.  All Rights Reserved.
:license: https://www.apache.org/licenses/LICENSE-2.0.html
"""
import re

import h5py
import numpy
from pykern import pkcli
from pykern import pkio
from pykern import pksubprocess
from pykern.pkcollections import PKDict
from pykern.pkdebug import pkdlog

#: a placed-element line, e.g. `"D001#0": "D001",elemedge=0.081058;`
_PLACEMENT_RE = re.compile(
    r'^\s*"([A-Za-z0-9_#]+)"\s*:\s*"([A-Za-z0-9_]+)"\s*,\s*elemedge\s*=\s*([-+0-9.eE]+)\s*;'
)

#: an element/command type definition, e.g. `"CAV010": RFCAVITY,l=0.25,...;`
_DEF_RE = re.compile(r'^\s*"([A-Za-z0-9_]+)"\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*,(.*);\s*$')

#: the start of a beamline LINE statement, e.g. `BL1: LINE=("D001#0",...);`
_LINE_START_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*LINE\s*=\s*\(")

#: strips the "#0"-style placement suffix off a placed element name
_PLACED_SUFFIX_RE = re.compile(r"#\d+$")

_LENGTH_RE = re.compile(r"(?:^|,)\s*l\s*=\s*([-+0-9.eE]+)")
_FMAPFN_RE = re.compile(r'fmapfn\s*=\s*"([^"]+)"')
_FNAME_RE = re.compile(r'fname\s*=\s*"([^"]+)"')


def trim_lattice(input_file, start, end, output_dir=None, opal_bin="opal", mpi_ranks=None):
    """Split ``input_file``'s beamline into an independent sub-lattice
    running from element ``start`` through element ``end``, using a real
    tracked beam (not the original distribution) as its input.

    Runs ``input_file`` once, truncated at ``start``, to capture the actual
    beam state there, converts it to an OPAL FROMFILE distribution, and
    writes a second, self-contained lattice file spanning ``start``..``end``
    that uses it -- so downstream studies can skip re-simulating everything
    upstream of ``start`` on every iteration.

    Known limitation: restarting from a single beam snapshot cannot carry
    over whatever sub-step integrator state (e.g. leapfrog half-step
    momenta) a continuous run would have, so this is not bit-identical to a
    single unbroken run. Bulk quantities (energy, particle count) match a
    continuous run to ~1e-4 relative in testing, but second-moment
    quantities (rms size, emittance) can differ by several percent,
    especially over short, early, space-charge-dominated stretches close
    to the source -- a 5.9 m section (D001->D020) showed 2-8% emittance/rms
    differences between a direct run and one chained through an
    intermediate restart, vs <1% for an 11 m stretch further downstream
    (CAV010->CAV030). Verify against a continuous baseline run before
    trusting a trim for a study that specifically targets an early,
    high-gradient section.

    Args:
        input_file (str): OPAL lattice file to trim (e.g. template.i)
        start (str): element name the trimmed lattice begins at
        end (str): element name the trimmed lattice ends at
        output_dir (str): where to write results (default: ``<start>_to_<end>``
            next to ``input_file``)
        opal_bin (str): OPAL executable (default "opal")
        mpi_ranks (int): if given, run the precompute step via
            ``mpirun -np mpi_ranks`` instead of serial ``opal``

    Returns:
        py.path.local: output_dir
    """
    ctx = PKDict(
        input_file=pkio.py_path(input_file),
        start=start,
        end=end,
        opal_bin=opal_bin,
        mpi_ranks=mpi_ranks,
    )
    ctx.input_dir = ctx.input_file.dirpath()
    ctx.lines = ctx.input_file.readlines()

    _, ctx.placement_order, ctx.line_first, ctx.line_last = _parse_line_statement(ctx.lines)
    ctx.placements = _parse_placements(ctx.lines)
    ctx.type_defs = _parse_type_defs(ctx.lines)

    base_to_index = {}
    for idx, placed_name in enumerate(ctx.placement_order):
        base_to_index.setdefault(_base_name(placed_name), idx)

    ctx.start_idx = _find_index(base_to_index, start, "start")
    ctx.end_idx = _find_index(base_to_index, end, "end")
    if ctx.start_idx > ctx.end_idx:
        pkcli.command_error("start element {} comes after end element {}", start, end)

    if output_dir is None:
        output_dir = ctx.input_dir.join(f"{start}_to_{end}")
    ctx.output_dir = pkio.mkdir_parent(output_dir)

    ctx.start_edge = ctx.placements[ctx.placement_order[ctx.start_idx]][1]
    end_base = _base_name(ctx.placement_order[ctx.end_idx])
    end_length = (ctx.type_defs.get(end_base) or {}).get("length") or 0.0
    ctx.end_edge = ctx.placements[ctx.placement_order[ctx.end_idx]][1] + end_length

    # adds n_particles, beam_file, pc, energy
    _run_precompute(ctx)

    _write_trimmed_lattice(ctx)

    pkdlog("wrote trimmed lattice to {}", ctx.output_dir)
    return ctx.output_dir


def _base_name(placed_name):
    return _PLACED_SUFFIX_RE.sub("", placed_name).upper()


def _find_command_line(lines, keyword):
    pattern = re.compile(rf'^\s*"([A-Za-z0-9_]+)"\s*:\s*{keyword}\s*,', re.IGNORECASE)
    for i, line in enumerate(lines):
        if pattern.match(line):
            return i
    pkcli.command_error("no {} statement found in lattice", keyword)


def _find_index(base_to_index, name, role):
    key = name.upper()
    if key not in base_to_index:
        pkcli.command_error(
            "{} element {} not found in lattice (have you got the right name? "
            "e.g. {})",
            role,
            name,
            ", ".join(sorted(base_to_index)[:10]),
        )
    return base_to_index[key]


def _parse_line_statement(lines):
    """Find the beamline LINE=(...) statement (it may wrap over several
    physical lines) and return (name, ordered placed-element names,
    first_line_idx, last_line_idx).
    """
    for i, line in enumerate(lines):
        m = _LINE_START_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        j = i
        text = line
        while ");" not in text:
            j += 1
            text += lines[j]
        names = re.findall(r'"([^"]+)"', text[text.index("(") :])
        return name, names, i, j
    pkcli.command_error("no LINE statement found in lattice")


def _parse_placements(lines):
    """Return {placed_name: (base_name, elemedge, line_idx)}."""
    result = {}
    for i, line in enumerate(lines):
        m = _PLACEMENT_RE.match(line)
        if m:
            result[m.group(1)] = (m.group(2), float(m.group(3)), i)
    return result


def _parse_type_defs(lines):
    """Return {base_name: {"length": float|None, "fmapfn": str|None}}."""
    result = {}
    for line in lines:
        if _PLACEMENT_RE.match(line):
            continue
        m = _DEF_RE.match(line)
        if not m:
            continue
        attrs = m.group(3)
        length_m = _LENGTH_RE.search(attrs)
        fmapfn_m = _FMAPFN_RE.search(attrs)
        result[m.group(1).upper()] = {
            "length": float(length_m.group(1)) if length_m else None,
            "fmapfn": fmapfn_m.group(1) if fmapfn_m else None,
        }
    return result


def _read_final_beam(h5_path):
    """Read the last tracked step of an opal.h5 file.

    Returns (x, px, y, py, z, pz) arrays -- x/y/z in meters, px/py/pz in
    beta*gamma, same convention as `load_opal_final` in
    newton/compare2/full-lattice/space-charge-25mA/sim_io.py.
    """
    with h5py.File(str(h5_path), "r") as f:
        steps = sorted(int(k.split("#")[1]) for k in f.keys() if k.startswith("Step#"))
        g = f[f"Step#{steps[-1]}"]
        return (
            g["x"][:],
            g["px"][:],
            g["y"][:],
            g["py"][:],
            g["z"][:],
            g["pz"][:],
        )


def _run_opal(work_dir, lattice_filename, opal_bin, mpi_ranks):
    cmd = [opal_bin, lattice_filename]
    if mpi_ranks:
        cmd = ["mpirun", "-np", str(mpi_ranks)] + cmd
    with pkio.save_chdir(work_dir):
        pksubprocess.check_call_with_signals(cmd, output="run.log", msg=pkdlog)


def _run_precompute(ctx):
    """Run the untouched lattice, truncated at `ctx.start_edge`, to capture
    the real beam state there. Adds `n_particles`, `beam_file`, `pc`,
    `energy` to `ctx`.
    """
    work_dir = pkio.mkdir_parent(ctx.output_dir.join("precompute"))

    track_idx = _find_command_line(ctx.lines, "track")
    new_lines = list(ctx.lines)
    new_lines[track_idx] = _set_attr(new_lines[track_idx], "zstop", ctx.start_edge)

    lattice_name = ctx.input_file.basename
    work_dir.join(lattice_name).write("".join(new_lines))

    referenced = {info["fmapfn"] for info in ctx.type_defs.values() if info["fmapfn"]}
    dist_idx = _find_command_line(ctx.lines, "distribution")
    m = _FNAME_RE.search(ctx.lines[dist_idx])
    if m:
        referenced.add(m.group(1))
    _symlink_referenced_files(ctx.input_dir, work_dir, referenced)

    _run_opal(work_dir, lattice_name, ctx.opal_bin, ctx.mpi_ranks)

    h5_name = re.sub(r"\.[^.]+$", "", lattice_name) + ".h5"
    x, px, y, py, z, pz = _read_final_beam(work_dir.join(h5_name))
    ctx.n_particles = len(x)
    ctx.beam_file = ctx.output_dir.join(f"beam_{ctx.start}.txt")
    _write_fromfile_beam(ctx.beam_file, x, px, y, py, z, pz)

    # OPAL's FROMFILE reader requires the BEAM command's reference
    # energy/momentum to match the distribution being loaded (it derives
    # its own reference momentum from `energy`+`mass`, not from the
    # `pc` attribute, and errors out -- "Distribution::checkFileMomentum"
    # -- if they disagree). Re-derive both from this beam's mean pz
    # (beta*gamma) and the original mass.
    beam_idx = _find_command_line(ctx.lines, "beam")
    mass_m = re.search(r"mass\s*=\s*([-+0-9.eE]+)", ctx.lines[beam_idx])
    mass = float(mass_m.group(1))
    beta_gamma = float(numpy.mean(pz))
    ctx.pc = beta_gamma * mass
    ctx.energy = mass * (1.0 + beta_gamma**2) ** 0.5

    pkdlog("precompute: {} particles survived to element {}", ctx.n_particles, ctx.start)


def _set_attr(line, attr, value):
    """Replace ``attr=...`` in an OPAL statement line -- same idiom as
    scan_failures_v2.py's `modify_for_failure` uses for VOLT.
    """
    pattern = re.compile(rf"({re.escape(attr)}\s*=\s*)[^,;]+", re.IGNORECASE)
    new_line, count = pattern.subn(lambda m: m.group(1) + str(value), line, count=1)
    if count == 0:
        pkcli.command_error("attribute {} not found in line: {}", attr, line)
    return new_line


def _symlink_referenced_files(input_dir, work_dir, filenames):
    for f in filenames:
        if not f:
            continue
        dst = work_dir.join(f)
        if not dst.exists():
            dst.mksymlinkto(input_dir.join(f))


def _write_fromfile_beam(path, x, px, y, py, z, pz):
    """Write an OPAL FROMFILE distribution file: header = particle count,
    then rows of "x px y py z pz" -- same convention as
    `_write_track_distribution` in newton/compare2's track-conversion.py
    (x/y/z in meters, px/py/pz in beta*gamma, no unit conversion needed
    going from opal.h5 since inputmounits="NONE" expects exactly this).
    """
    output = numpy.column_stack((x, px, y, py, z, pz))
    with open(str(path), "w") as f:
        f.write(f"{len(output)}\n")
        numpy.savetxt(f, output, fmt="%.12e")


def _write_trimmed_lattice(ctx):
    beam_idx = _find_command_line(ctx.lines, "beam")
    dist_idx = _find_command_line(ctx.lines, "distribution")
    track_idx = _find_command_line(ctx.lines, "track")

    kept_order = ctx.placement_order[ctx.start_idx : ctx.end_idx + 1]
    kept_names = set(kept_order)
    kept_bases = {_base_name(n) for n in kept_names}

    new_lines = []
    for i, line in enumerate(ctx.lines):
        if ctx.line_first <= i <= ctx.line_last:
            if i == ctx.line_first:
                new_lines.append(
                    "BL2: LINE=({});\n".format(",".join(f'"{n}"' for n in kept_order))
                )
            continue

        m = _PLACEMENT_RE.match(line)
        if m:
            placed_name = m.group(1)
            if placed_name not in kept_names:
                continue
            base_name, edge, _ = ctx.placements[placed_name]
            new_edge = edge - ctx.start_edge
            new_lines.append(f'"{placed_name}": "{base_name}",elemedge={new_edge:.9g};\n')
            continue

        if i == beam_idx:
            line = _set_attr(line, "npart", f"{ctx.n_particles}.0")
            line = _set_attr(line, "pc", f"{ctx.pc:.9g}")
            line = _set_attr(line, "energy", f"{ctx.energy:.9g}")
        elif i == dist_idx:
            line = _set_attr(line, "fname", f'"{ctx.beam_file.basename}"')
        elif i == track_idx:
            line = _set_attr(line, "line", "BL2")
            line = _set_attr(line, "zstop", f"{ctx.end_edge - ctx.start_edge:.9g}")
        new_lines.append(line)

    out_lattice = ctx.output_dir.join(f"{ctx.start}_to_{ctx.end}.i")
    out_lattice.write("".join(new_lines))

    needed_fmapfn = {
        ctx.type_defs[b]["fmapfn"]
        for b in kept_bases
        if (ctx.type_defs.get(b) or {}).get("fmapfn")
    }
    _symlink_referenced_files(ctx.input_dir, ctx.output_dir, needed_fmapfn)
