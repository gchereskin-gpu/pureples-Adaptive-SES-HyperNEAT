"""
Helpers for multi-run data collection across the adaptive ES/DES T-Maze studies.

Used when an experiment's ``NUM_RUNS`` is greater than 1. EVERY run -- successful
or not -- records its per-generation fitness, a handful of network statistics,
and the four post-evolution test results measured on its best genome; once every
run has finished, a summary fitness graph and a summary text file aggregate them.

The summary reports each solution-quality statistic twice: over all saved runs
and over the successful runs alone. Because unsuccessful runs are now tested too,
the all-runs figures describe what the algorithm typically produces, rather than
only what it produces on the runs that happened to clear the fitness threshold.
"""

import os
import json
import statistics as _stats
from dataclasses import dataclass, asdict, fields
from typing import List

import matplotlib.pyplot as plt

from pureples.experiments.t_maze.multi_t_maze import EXPECTED_ADAPTIVE_FITNESS


@dataclass
class RunRecord:
    """
    Metrics collected from a single evolutionary run.

    Every run -- successful or not -- records its per-generation fitness series
    plus the solution-quality fields below, all measured on the run's best
    genome. They stay ``None`` only where the metric does not apply
    (``num_branches`` is Adaptive DES only) or in records written by an older
    version of the study, which measured them for successful runs alone.
    """
    run_number: int
    generations_to_solution: int        # generations recorded; == the budget for an unsolved run
    mean_fitness_per_gen: List[float]   # padded to the generation budget
    best_fitness_per_gen: List[float]   # padded to the generation budget
    solved: bool = True                 # reached the fitness threshold within the budget
    # Solution-quality metrics, measured on the run's best genome (see above).
    mean_an_product: float = None       # mean of A*n over standard connections
    inhibitory_count: int = None        # # std connections with A*n < 0
    excitatory_count: int = None        # # std connections with A*n > 0
    cppn_connection_count: int = None   # enabled connections in the evolved CPPN genome
    cppn_node_count: int = None         # input + hidden + output nodes of the CPPN
    num_branches: int = None            # # branches (Adaptive DES only; None for ES)
    # Size of the phenotype network the CPPN expands into. std_connection_count
    # is the total, so it also covers the connections with A*n exactly 0 that
    # neither inhibitory_count nor excitatory_count includes.
    network_node_count: int = None      # nodes in the phenotype (inputs + hidden + outputs)
    std_connection_count: int = None    # standard connections in the phenotype
    mod_connection_count: int = None    # modulatory connections that affect the phenotype
    # Post-evolution test means (see multi_t_maze). None until measured.
    all_switch_fitness: float = None       # Test 1: mean over all training-range single-switch schedules
    double_switch_fitness: float = None    # Test 2: mean over the double-switch generalization deployments
    delayed_feedback_fitness: float = None  # Test 3: mean over the delayed-feedback deployments
    constant_feedback_fitness: float = None  # Test 4: mean over the constant-feedback deployments

    @property
    def connection_ratio(self):
        """
        Inhibitory/excitatory connection ratio (inf when no excitatory
        connections), or None when the connection counts were never measured
        (a record written before every run was analysed).
        """
        if self.inhibitory_count is None or self.excitatory_count is None:
            return None
        return _safe_ratio(self.inhibitory_count, self.excitatory_count)


def save_run_record(path, record):
    """
    Persist ``record`` to ``path`` as JSON so a later batch can reload it.

    The summary written after each batch reloads every run's record from disk
    (see load_saved_records), so a study can be built up over several separate
    invocations without losing earlier runs' data.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(record), f, indent=2)


def load_run_record(path):
    """Reload a RunRecord previously written by save_run_record."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    known = {field.name for field in fields(RunRecord)}
    return RunRecord(**{k: v for k, v in data.items() if k in known})


def run_record_path(data_dir, base_name, run_number):
    """Path of the persisted JSON record for a single run."""
    return os.path.join(data_dir, f"{base_name}_run{run_number}.json")


def load_saved_records(data_dir, base_name, end_run):
    """
    Load every run record currently saved on disk for runs 1..``end_run``.

    Missing run numbers (never run, or an unsolved run in an old-style study that
    saved nothing) are simply skipped, so the summary covers whatever data exists
    -- exactly what lets a study accumulate across batches. Records are returned
    ordered by run number.
    """
    records = []
    for run_number in range(1, end_run + 1):
        path = run_record_path(data_dir, base_name, run_number)
        if os.path.exists(path):
            records.append(load_run_record(path))
    records.sort(key=lambda r: r.run_number)
    return records


def _safe_ratio(numerator, denominator):
    """``numerator/denominator``, or ``float('inf')`` when the denominator is 0."""
    return numerator / denominator if denominator else float("inf")


def _mean_std(values):
    """
    Return ``(mean, std dev, n)`` over the values that are present.

    ``None`` entries are dropped, so a metric that some records never measured
    still reports a meaningful figure alongside the count it was taken over.
    Returns ``(None, None, 0)`` when nothing is left, and a std dev of 0.0 for a
    single value.
    """
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    return (_stats.mean(values),
            _stats.stdev(values) if len(values) > 1 else 0.0,
            len(values))


def solved(winner, config):
    """
    Return True when ``winner`` reached the configured fitness threshold.

    Mirrors neat's own termination test: a run is successful only when the best
    genome's fitness meets ``fitness_threshold`` (runs that exhaust the
    generation budget without doing so end with a below-threshold best genome).
    """
    return winner.fitness is not None and winner.fitness >= config.fitness_threshold


def padded_fitness_series(stats, max_generations):
    """
    Return ``(mean_per_gen, best_per_gen)`` for one run, each padded to
    ``max_generations`` entries.

    A run that solves the task in G < ``max_generations`` generations only has G
    recorded generations; the remaining entries are filled with the final
    generation's value so every run contributes a full-length series.
    """
    mean_per_gen = list(stats.get_fitness_mean())
    best_per_gen = [g.fitness for g in stats.most_fit_genomes]

    def pad(series):
        if not series:
            return [0.0] * max_generations
        if len(series) >= max_generations:
            return series[:max_generations]
        return series + [series[-1]] * (max_generations - len(series))

    return pad(mean_per_gen), pad(best_per_gen)


def connection_sign_metrics(an_products):
    """
    Given the A*n product for every standard connection, return
    ``(mean_product, inhibitory_count, excitatory_count)``.

    A connection is inhibitory when A*n < 0 and excitatory when A*n > 0
    (products of exactly 0 are counted as neither).
    """
    products = list(an_products)
    mean_product = sum(products) / len(products) if products else 0.0
    inhibitory = sum(1 for p in products if p < 0.0)
    excitatory = sum(1 for p in products if p > 0.0)
    return mean_product, inhibitory, excitatory


def count_cppn_connections(genome):
    """Number of expressed (enabled) connections in the evolved CPPN genome."""
    return sum(1 for cg in genome.connections.values() if cg.enabled)


def count_cppn_nodes(genome, config):
    """
    Number of nodes in the evolved CPPN genome: inputs + hidden + outputs.

    neat only stores node genes for the hidden and output nodes (the inputs are
    implicit, one per ``genome_config.input_keys`` entry), so the inputs are
    added back here to give the size of the CPPN as drawn.
    """
    return len(genome.nodes) + len(config.genome_config.input_keys)


def write_run_report(path, record, max_generations, title):
    """
    Write the per-run text report: the generations-to-solution count (successful
    runs only), the four post-evolution test means, the two A*n ratios, the CPPN
    connection/node counts, and the padded per-generation mean/best fitness
    series. Everything except the generations-to-solution line is written for
    every run, successful or not, since every run's best genome is tested.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title} — run {record.run_number} report\n")
        f.write("=" * 60 + "\n\n")

        if record.solved:
            f.write(f"Generations to reach fitness threshold: {record.generations_to_solution}\n\n")
        else:
            f.write("Result: did NOT reach the fitness threshold within "
                    f"{record.generations_to_solution} generations.\n")
            f.write("  (The run's best genome was still analysed and put through the four\n"
                    "   post-evolution tests below, so it contributes to the summary graph\n"
                    "   and to the all-runs statistics in the summary file.)\n\n")

        if (record.all_switch_fitness is not None
                or record.double_switch_fitness is not None
                or record.delayed_feedback_fitness is not None
                or record.constant_feedback_fitness is not None):
            f.write("Post-evolution tests (mean fitness per deployment; not seen in training):\n")
            if record.all_switch_fitness is not None:
                f.write(f"  Test 1 -- all training-range single-switch schedules: "
                        f"{record.all_switch_fitness:.6f}\n")
            if record.double_switch_fitness is not None:
                f.write(f"  Test 2 -- double-switch generalization (20 episodes): "
                        f"{record.double_switch_fitness:.6f}\n")
            if record.delayed_feedback_fitness is not None:
                f.write(f"  Test 3 -- delayed reward feedback: "
                        f"{record.delayed_feedback_fitness:.6f}\n")
            if record.constant_feedback_fitness is not None:
                f.write(f"  Test 4 -- constant (uninformative) reward feedback: "
                        f"{record.constant_feedback_fitness:.6f}\n")
            f.write("\n")

        if (record.network_node_count is not None
                or record.std_connection_count is not None
                or record.num_branches is not None):
            f.write("Final network size (the phenotype the CPPN expands into):\n")
            if record.network_node_count is not None:
                f.write(f"  Nodes: {record.network_node_count}\n")
            if record.std_connection_count is not None:
                f.write(f"  Standard connections: {record.std_connection_count}\n")
            if record.mod_connection_count is not None:
                f.write(f"  Modulatory connections: {record.mod_connection_count}\n")
            if record.num_branches is not None:
                f.write(f"  Branches: {record.num_branches}\n")
            f.write("\n")

        if record.mean_an_product is not None:
            ratio = record.connection_ratio
            ratio_text = ("undefined (no excitatory connections)"
                          if ratio == float("inf") else f"{ratio:.6f}")
            f.write("Excitatory/inhibitory analysis (sign of A*n per standard connection):\n")
            f.write(f"  Inhibition/excitation ratio (mean of A*n over standard connections): "
                    f"{record.mean_an_product:.6f}\n")
            f.write(f"  Inhibitory/excitatory connection ratio: {ratio_text}\n")
            f.write(f"    inhibitory (A*n < 0) standard connections: {record.inhibitory_count}\n")
            f.write(f"    excitatory (A*n > 0) standard connections: {record.excitatory_count}\n\n")

        if (record.cppn_connection_count is not None
                or record.cppn_node_count is not None):
            f.write("Evolved CPPN genome size:\n")
            if record.cppn_connection_count is not None:
                f.write(f"  Enabled connections: {record.cppn_connection_count}\n")
            if record.cppn_node_count is not None:
                f.write(f"  Nodes: {record.cppn_node_count}\n")
            f.write("\n")

        f.write(f"Per-generation fitness (padded to {max_generations} generations):\n")
        f.write("generation,mean_fitness,best_fitness\n")
        for gen in range(max_generations):
            mean_v = record.mean_fitness_per_gen[gen]
            best_v = record.best_fitness_per_gen[gen]
            f.write(f"{gen},{mean_v:.6f},{best_v:.6f}\n")


def write_summary(graph_path, text_path, records, total_runs, max_generations, title):
    """
    Write the across-runs summary graph and text file.

    ``records`` is every run saved on disk so far (see load_saved_records),
    successful and unsuccessful alike.

      * The per-generation fitness curves -- the summary graph and the
        per-generation table -- average over ALL runs, so an unsuccessful run
        drags the curve down and the summary reflects the true success rate.
      * Each solution-quality statistic (post-evolution tests, A*n ratios, CPPN
        size/branches) is reported twice, with mean and standard deviation: once
        over ALL saved runs and once over the SUCCESSFUL runs alone. Every run's
        best genome is tested, so the all-runs figures measure what the algorithm
        typically produces, while the successful-only figures measure what it
        produces on the runs that cleared the fitness threshold. Comparing the
        two shows how much the threshold biases the successful-only view.
      * Generations-to-solution stays successful-only: an unsuccessful run simply
        used the whole budget, so averaging it across all runs is meaningless.

    Each statistic is reported alongside the number of runs it was taken over,
    which may be below the group size for records written before every run was
    analysed (those left the fields unset).

    The graph plots, per generation, the mean over runs of each run's mean
    fitness (line) and the mean over runs of each run's best fitness (dots); no
    standard deviation is drawn.
    """
    if not records:
        return

    successful = [r for r in records if r.solved]

    generations = list(range(max_generations))
    mean_of_means = [
        _stats.mean(r.mean_fitness_per_gen[g] for r in records) for g in generations
    ]
    mean_of_bests = [
        _stats.mean(r.best_fitness_per_gen[g] for r in records) for g in generations
    ]

    # --- Summary graph (all runs) -----------------------------------------
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(generations, mean_of_means, color="tab:blue", linewidth=1.5,
            label="Mean of per-run mean fitness")
    ax.scatter(generations, mean_of_bests, color="tab:red", s=16, zorder=3,
               label="Mean of per-run best fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(f"{title} — fitness over {len(records)} run(s) "
                 f"({len(successful)} solved)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(graph_path, bbox_inches="tight")
    plt.close(fig)

    # --- Solution-quality statistics ---------------------------------------
    # Every metric is reported for both groups; see this function's docstring.
    groups = (("all runs", records), ("successful", successful))

    gens_mean, gens_std, gens_n = _mean_std(
        r.generations_to_solution for r in successful)

    def finite_connection_ratio(record):
        """
        A record's inhibitory/excitatory connection ratio, or None when it is
        unmeasured or infinite. Infinite ratios (a network with no excitatory
        connections at all) would swamp the mean, so they are excluded and
        counted separately in the report.
        """
        ratio = record.connection_ratio
        return None if ratio in (None, float("inf")) else ratio

    # Each test also carries the key of its EXPECTED_ADAPTIVE_FITNESS entry, so
    # the summary prints what an ideal adaptive agent would score next to what was
    # actually measured. The network metrics below have no such reference and pass
    # None.
    test_metrics = [
        ("Test 1 -- all training-range single-switch schedules",
         lambda r: r.all_switch_fitness, "{:.4f}", "all_switch"),
        ("Test 2 -- double-switch generalization (20 episodes)",
         lambda r: r.double_switch_fitness, "{:.4f}", "double_switch"),
        ("Test 3 -- delayed reward feedback",
         lambda r: r.delayed_feedback_fitness, "{:.4f}", "delayed_feedback"),
        ("Test 4 -- constant (uninformative) reward feedback",
         lambda r: r.constant_feedback_fitness, "{:.4f}", "constant_feedback"),
    ]

    # Grouped: how big the final network is, how its connections are signed, and
    # how big the CPPN genome that generated it is.
    network_metrics = [
        ("Nodes in the final network",
         lambda r: r.network_node_count, "{:.4f}"),
        ("Standard connections in the final network",
         lambda r: r.std_connection_count, "{:.4f}"),
        ("Modulatory connections in the final network",
         lambda r: r.mod_connection_count, "{:.4f}"),
        ("Branches in the final network",
         lambda r: r.num_branches, "{:.4f}"),
        ("Mean A*n over standard connections (inhibition/excitation ratio)",
         lambda r: r.mean_an_product, "{:.6f}"),
        ("Inhibitory/excitatory connection ratio",
         finite_connection_ratio, "{:.6f}"),
        ("Inhibitory (A*n < 0) standard connections",
         lambda r: r.inhibitory_count, "{:.4f}"),
        ("Excitatory (A*n > 0) standard connections",
         lambda r: r.excitatory_count, "{:.4f}"),
        ("Enabled connections in the evolved CPPN genome",
         lambda r: r.cppn_connection_count, "{:.4f}"),
        ("Nodes in the evolved CPPN genome",
         lambda r: r.cppn_node_count, "{:.4f}"),
    ]

    # --- Summary text ------------------------------------------------------
    os.makedirs(os.path.dirname(text_path), exist_ok=True)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(f"{title} — summary over all saved runs\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Runs with saved data: {len(records)} of {total_runs} expected "
                f"(runs 1..{total_runs})\n")
        f.write(f"Successful (solved) runs: {len(successful)}\n")
        f.write("  Fitness curves (graph + per-generation table) average over ALL "
                "saved runs.\n  Every statistic below is given twice: over all saved "
                "runs, and over the\n  successful runs alone. Every run's best genome "
                "is tested, so the all-runs\n  figures are not conditioned on reaching "
                "the fitness threshold.\n\n")

        def write_metric(label, getter, fmt, reference_key=None):
            """
            Write one metric's mean/std dev for both groups, or skip it entirely.

            ``reference_key`` names an EXPECTED_ADAPTIVE_FITNESS entry; when given,
            the expected/optimum/ceiling fitnesses are printed above the measured
            figures so the two can be read side by side.
            """
            stats_by_group = [(name, _mean_std(getter(r) for r in group))
                              for name, group in groups]
            if all(n == 0 for _, (_, _, n) in stats_by_group):
                return  # never measured (e.g. branches on an Adaptive ES study)
            f.write(f"  {label}:\n")
            if reference_key is not None:
                expected, optimum, ceiling = EXPECTED_ADAPTIVE_FITNESS[reference_key]
                f.write(f"    {'expected':<11} (win-stay/lose-shift): "
                        f"{fmt.format(expected)}"
                        f"   optimum = {fmt.format(optimum)}"
                        f"   ceiling = {fmt.format(ceiling)}\n")
            for name, (mean_v, std_v, n) in stats_by_group:
                if mean_v is None:
                    f.write(f"    {name:<11} (n={n:>3}): no data\n")
                else:
                    f.write(f"    {name:<11} (n={n:>3}): mean = {fmt.format(mean_v)}"
                            f"   std dev = {fmt.format(std_v)}\n")

        if successful:
            f.write("Generations to find a solution (successful runs only -- an "
                    "unsuccessful run\nsimply used the whole budget):\n")
            f.write(f"    successful  (n={gens_n:>3}): mean = {gens_mean:.4f}"
                    f"   std dev = {gens_std:.4f}\n\n")
        else:
            f.write("No successful runs saved yet, so no generations-to-solution "
                    "statistic.\n\n")

        f.write("Post-evolution test fitness (per-run mean over that test's "
                "deployments;\nmeasured on each run's best genome and never seen in "
                "training):\n")
        f.write("  Each test lists three exact reference fitnesses above its measured\n"
                "  figures, all on the same per-deployment-mean scale as the results:\n"
                "    expected -- what the natural adaptive strategy (win-stay/lose-shift,\n"
                "                probing the same arm first every deployment) scores.\n"
                "                This is the figure an adaptive network is aiming at.\n"
                "    optimum  -- the best of ALL deterministic reward-history policies;\n"
                "                it beats win-stay/lose-shift only by exploiting the\n"
                "                bounded switch window, so a measured score at or above\n"
                "                it means something is wrong with the measurement.\n"
                "    ceiling  -- a clairvoyant agent that never needs a probe episode;\n"
                "                unreachable by anything that must detect switches.\n"
                "  See EXPECTED_ADAPTIVE_FITNESS in multi_t_maze for each derivation.\n"
                "  Note Test 4's expected value is 0.0: with the feedback pinned to 1.0 a\n"
                "  purely reward-driven agent never sees a loss, never switches arms, and\n"
                "  is zeroed by the single-arm rule. A Test 4 score well above 0 therefore\n"
                "  indicates arm switching driven by something other than reward content\n"
                "  (an internal timer or oscillation), not adaptation.\n")
        for label, getter, fmt, reference_key in test_metrics:
            write_metric(label, getter, fmt, reference_key)
        f.write("\n")

        f.write("Network analysis (measured on each run's best genome):\n")
        for label, getter, fmt in network_metrics:
            write_metric(label, getter, fmt)
            if getter is finite_connection_ratio:
                # Note the infinite ratios dropped by finite_connection_ratio, so
                # the n above is not mistaken for missing data.
                for name, group in groups:
                    excluded = sum(1 for r in group
                                   if r.connection_ratio == float("inf"))
                    if excluded:
                        f.write(f"      ({name}: {excluded} run(s) excluded -- no "
                                f"excitatory connections, ratio is infinite)\n")
        f.write("\n")

        f.write("Per-generation summary fitness (mean over ALL saved runs; "
                "matches the summary graph):\n")
        f.write("generation,mean_of_mean_fitness,mean_of_best_fitness\n")
        for gen in generations:
            f.write(f"{gen},{mean_of_means[gen]:.6f},{mean_of_bests[gen]:.6f}\n")
