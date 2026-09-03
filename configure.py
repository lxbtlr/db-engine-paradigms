#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p python3 python3Packages.questionary python3Packages.rich

"""
Interactive TUI for configuring and building the allocator project.
Presents CMake options via questionary, runs cmake + make, and shows
a progress bar during build. On failure, displays captured output.
"""

import os
import sys
import subprocess
import threading
import time
import questionary
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
console = Console()

# ── CMake option definitions ──────────────────────────────────────────────

BOOL_OPTIONS = [
    ("USE_CLANG",             "Use clang-18/clang++-18 instead of gcc/g++",              False),
    ("AUTOVECTORIZE",         "Allow the compiler to autovectorize",                      False),
    ("SCAN_STATIC_PARTITION", "Static per-thread slice partitioning in Scan::next()",     False),
    ("ORIGINAL_GROUPLOOKUP",  "Use original single-pass GroupLookup (no prefetch split)", False),
    ("NUMA_ALLOC",            "Replicate base-table columns per NUMA region (4x copy)",   False),
    ("NUMA_SHARD",            "Shard base-table columns across NUMA regions",             False),
    ("THREAD_PIN_PACKED",     "Pack threads per socket (vs spread round-robin)",           False),
    ("NUMA_DEBUG",            "NUMA placement verification (move_pages)",                  False),
    ("USE_TCMALLOC",          "Use tcmalloc from gperftools as the allocator",             False),
    ("VW_USE_CRC32",          "Use CRC32 hashing for VectorWise (default: MurMurHash)",   False),
    ("VECTORWISE_BRANCHING",  "Use branching vectorwise primitives",                       False),
    ("AVX512EXPERIMENTS",     "Enable AVX512 experiment targets",                          False),
    ("HARDWARE_BENCHMARKS",   "Enable hardware benchmark targets",                         False),
    ("NO_HUGE_PAGES",         "Disable huge pages for malloc_huge",                        False),
    ("HUGE_2MB_MALLOC_HUGE",  "Use 2MB hugetlb pages for malloc_huge",                    False),
    ("HUGE_1GB_MALLOC_HUGE",  "Use 1GB huge pages for malloc_huge",                       False),
    ("HUGE_1GB_MALLOC_NUMA",  "Use 1GB huge pages for malloc_numa",                       False),
    ("WARM_PAGES",            "Touch all relation pages before measurement",               False),
    ("INTERLEAVE_HT",         "Interleave hash table pages across NUMA nodes",             False),
    ("HYPER_FLOAT",           "Let TBB hyper-engine threads float (no pinning)",           False),
]

TARGET_ARCH_CHOICES = [
    "native",
    "skylake-x",
    "icelake",
    "sapphirerapids",
    "neoverse-n1",
    "neoverse-v1",
    "neoverse-v2",
    "ampere-altra",
    "armv8.2-generic",
    "armv9-generic",
]

BUILD_TYPES = ["Release", "RelWithDebInfo", "Debug"]

# ── Machine topology presets ──────────────────────────────────────────────
# Each entry: (SOCKETS_COUNT, CORES_PER_SOCKET, SMT_PER_CORE, THREAD_PIN_PACKED)

MACHINE_PRESETS = {
    "custom":    None,  # user sets values manually
    "dubliner":  (4, 22, 2, False),   # 4-socket Xeon, interleaved CPU numbering
    "roquefort": (1, 24, 2, False),   # 1-socket AMD EPYC 7443P (Zen3)
    "manchego":  (2, 8, 2, False),    # 2-socket Xeon Silver 4509Y (SPR), interleaved
    "burrata":   (1, 128, 1, False),  # 1-socket ARM Neoverse-N1
}

# ── Preset configurations ─────────────────────────────────────────────────

PRESETS = {
    "ARM baseline": {
        "USE_CLANG": True,
        "TARGET_ARCH": "neoverse-n1",
        "SCAN_STATIC_PARTITION": True,
    },
    "ARM optimized": {
        "USE_CLANG": True,
        "TARGET_ARCH": "neoverse-n1",
        "SCAN_STATIC_PARTITION": True,
        "AUTOVECTORIZE": True,
    },
    "ARM NUMA replicate": {
        "USE_CLANG": True,
        "TARGET_ARCH": "neoverse-n1",
        "SCAN_STATIC_PARTITION": True,
        "NUMA_ALLOC": True,
    },
    "x86 baseline": {
        "TARGET_ARCH": "native",
    },
    "x86 skylake AVX512": {
        "TARGET_ARCH": "skylake-x",
        "AVX512EXPERIMENTS": True,
    },
    "x86 NUMA shard": {
        "TARGET_ARCH": "native",
        "NUMA_SHARD": True,
    },
    "Custom": None,
}

# ── Validation ────────────────────────────────────────────────────────────

def validate_options(opts):
    errors = []
    if opts.get("NUMA_ALLOC") and opts.get("NUMA_SHARD"):
        errors.append("NUMA_ALLOC and NUMA_SHARD are mutually exclusive.")
    if opts.get("SCAN_STATIC_PARTITION") and opts.get("NUMA_SHARD"):
        errors.append("SCAN_STATIC_PARTITION and NUMA_SHARD are mutually exclusive.")
    return errors

# ── Interactive configuration ─────────────────────────────────────────────

def configure():
    console.print(Panel("[bold]Allocator Build Configuration[/bold]", style="cyan"))

    # Preset selection
    preset_name = questionary.select(
        "Select a configuration preset:",
        choices=list(PRESETS.keys()),
        default="Custom",
    ).ask()
    if preset_name is None:
        sys.exit(1)

    preset = PRESETS[preset_name]

    # Start with defaults
    opts = {name: default for name, _, default in BOOL_OPTIONS}
    opts["TARGET_ARCH"] = "native"
    opts["TARGET_MACHINE"] = "custom"
    opts["SOCKETS_COUNT"] = "4"
    opts["CORES_PER_SOCKET"] = "22"
    opts["SMT_PER_CORE"] = "2"
    opts["BUILD_TYPE"] = "Release"
    opts["DATADIR"] = ""
    opts["BUILD_DIR"] = "build"
    opts["JOBS"] = str(os.cpu_count() or 4)

    # Apply preset
    if preset is not None:
        for k, v in preset.items():
            opts[k] = v

    if preset_name == "Custom" or questionary.confirm(
        "Customize options further?", default=False
    ).ask():
        # Target architecture
        opts["TARGET_ARCH"] = questionary.select(
            "Target architecture:",
            choices=TARGET_ARCH_CHOICES,
            default=opts["TARGET_ARCH"],
        ).ask()
        if opts["TARGET_ARCH"] is None:
            sys.exit(1)

        # Build type
        opts["BUILD_TYPE"] = questionary.select(
            "Build type:",
            choices=BUILD_TYPES,
            default=opts["BUILD_TYPE"],
        ).ask()
        if opts["BUILD_TYPE"] is None:
            sys.exit(1)

        # Machine topology
        machine_name = questionary.select(
            "Machine topology preset:",
            choices=list(MACHINE_PRESETS.keys()),
            default=opts["TARGET_MACHINE"],
        ).ask()
        if machine_name is None:
            sys.exit(1)
        opts["TARGET_MACHINE"] = machine_name
        mp = MACHINE_PRESETS[machine_name]
        if mp is not None:
            opts["SOCKETS_COUNT"] = str(mp[0])
            opts["CORES_PER_SOCKET"] = str(mp[1])
            opts["SMT_PER_CORE"] = str(mp[2])
            opts["THREAD_PIN_PACKED"] = mp[3]
        else:
            opts["SOCKETS_COUNT"] = questionary.text(
                "Sockets / chiplets / memory domains:", default=opts["SOCKETS_COUNT"]).ask()
            opts["CORES_PER_SOCKET"] = questionary.text(
                "Cores per socket / chiplet:", default=opts["CORES_PER_SOCKET"]).ask()
            opts["SMT_PER_CORE"] = questionary.text(
                "Hardware threads per core (SMT):", default=opts["SMT_PER_CORE"]).ask()

        # Compiler & optimization group
        compiler_opts = questionary.checkbox(
            "Compiler & optimization options (space to toggle):",
            choices=[
                questionary.Choice(f"{name}: {desc}", value=name, checked=opts.get(name, default))
                for name, desc, default in BOOL_OPTIONS[:4]
            ],
        ).ask()
        if compiler_opts is None:
            sys.exit(1)
        for name, _, _ in BOOL_OPTIONS[:4]:
            opts[name] = name in compiler_opts

        # NUMA group
        numa_opts = questionary.checkbox(
            "NUMA options (space to toggle):",
            choices=[
                questionary.Choice(f"{name}: {desc}", value=name, checked=opts.get(name, default))
                for name, desc, default in BOOL_OPTIONS[4:11]
            ],
        ).ask()
        if numa_opts is None:
            sys.exit(1)
        for name, _, _ in BOOL_OPTIONS[4:11]:
            opts[name] = name in numa_opts

        # Advanced group
        show_advanced = questionary.confirm("Show advanced options?", default=False).ask()
        if show_advanced:
            adv_opts = questionary.checkbox(
                "Advanced options (space to toggle):",
                choices=[
                    questionary.Choice(f"{name}: {desc}", value=name, checked=opts.get(name, default))
                    for name, desc, default in BOOL_OPTIONS[11:]
                ],
            ).ask()
            if adv_opts is None:
                sys.exit(1)
            for name, _, _ in BOOL_OPTIONS[11:]:
                opts[name] = name in adv_opts

        # Data directory
        datadir = questionary.text(
            "Data directory (DATADIR, leave empty for default):",
            default=opts["DATADIR"],
        ).ask()
        if datadir is not None:
            opts["DATADIR"] = datadir

        # Build directory
        opts["BUILD_DIR"] = questionary.text(
            "Build directory (relative to project root):",
            default=opts["BUILD_DIR"],
        ).ask()
        if opts["BUILD_DIR"] is None:
            sys.exit(1)

        # Parallel jobs
        opts["JOBS"] = questionary.text(
            "Parallel build jobs (-j):",
            default=opts["JOBS"],
        ).ask()
        if opts["JOBS"] is None:
            sys.exit(1)

    # Validate
    errors = validate_options(opts)
    if errors:
        for e in errors:
            console.print(f"[red bold]Error:[/red bold] {e}")
        sys.exit(1)

    return opts


def build_cmake_args(opts):
    args = [f"-DCMAKE_BUILD_TYPE={opts['BUILD_TYPE']}"]
    args.append(f"-DTARGET_ARCH={opts['TARGET_ARCH']}")
    args.append(f"-DTARGET_MACHINE={opts['TARGET_MACHINE']}")
    if opts["TARGET_MACHINE"] == "custom":
        args.append(f"-DSOCKETS_COUNT={opts['SOCKETS_COUNT']}")
        args.append(f"-DCORES_PER_SOCKET={opts['CORES_PER_SOCKET']}")
        args.append(f"-DSMT_PER_CORE={opts['SMT_PER_CORE']}")
    if opts["DATADIR"]:
        args.append(f"-DDATADIR={opts['DATADIR']}")
    for name, _, _ in BOOL_OPTIONS:
        val = "ON" if opts.get(name, False) else "OFF"
        args.append(f"-D{name}={val}")
    return args

# ── Build execution ───────────────────────────────────────────────────────

def run_build(build_dir, cmake_args, jobs):
    abs_build = os.path.join(PROJECT_DIR, build_dir)
    os.makedirs(abs_build, exist_ok=True)

    # Show configuration summary
    summary = Text()
    summary.append("Build directory: ", style="bold")
    summary.append(f"{abs_build}\n")
    summary.append("CMake args:\n", style="bold")
    for arg in cmake_args:
        summary.append(f"  {arg}\n", style="dim")
    console.print(Panel(summary, title="Configuration Summary", style="green"))

    if not questionary.confirm("Proceed with build?", default=True).ask():
        console.print("[yellow]Build cancelled.[/yellow]")
        sys.exit(0)

    # ── CMake configure ──
    console.print("\n[bold cyan]Running cmake...[/bold cyan]")
    cmake_cmd = ["cmake"] + cmake_args + [PROJECT_DIR]
    result = subprocess.run(
        cmake_cmd,
        cwd=abs_build,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print("[red bold]CMake configuration failed![/red bold]")
        console.print(Panel(result.stdout + result.stderr, title="CMake Output", style="red"))
        sys.exit(1)
    console.print("[green]CMake configuration successful.[/green]")

    # ── Make build with progress ──
    console.print(f"\n[bold cyan]Building with make -j{jobs}...[/bold cyan]")

    proc = subprocess.Popen(
        ["make", f"-j{jobs}"],
        cwd=abs_build,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output_lines = []
    build_done = threading.Event()
    target_count = [0, 0]  # [current, total]

    def read_output():
        for line in proc.stdout:
            output_lines.append(line)
            # Parse make progress: [  5%] Building CXX ...
            if line.startswith("["):
                try:
                    pct = line.split("]")[0].strip("[ %")
                    target_count[0] = int(pct)
                    target_count[1] = 100
                except (ValueError, IndexError):
                    pass
        build_done.set()

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Building...", total=100)
        while not build_done.is_set():
            progress.update(task, completed=target_count[0])
            time.sleep(0.1)
        progress.update(task, completed=100)

    proc.wait()
    reader.join()

    if proc.returncode != 0:
        console.print("[red bold]Build failed![/red bold]")
        # Show last 50 lines of output for context
        fail_output = "".join(output_lines[-50:])
        console.print(Panel(fail_output, title="Build Output (last 50 lines)", style="red"))

        # Save full output
        log_path = os.path.join(abs_build, "build_error.log")
        with open(log_path, "w") as f:
            f.writelines(output_lines)
        console.print(f"[dim]Full build log saved to: {log_path}[/dim]")
        sys.exit(1)

    console.print("[green bold]Build successful![/green bold]")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    try:
        opts = configure()
        cmake_args = build_cmake_args(opts)
        run_build(opts["BUILD_DIR"], cmake_args, opts["JOBS"])
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(1)

if __name__ == "__main__":
    main()
