"""
main.py
=======
Master pipeline entry point for the Ship ESS Resilience Optimization project.

Execution modes (--mode flag)
------------------------------
  all      : Full end-to-end pipeline (default)
  data     : Generate input data files only
  gan      : Generate data + train GAN + generate scenarios
  surrogate: Above + train surrogate NN
  optimize : Above + run NSGA-II + decision making
  plot     : Reload saved results and regenerate all figures

Usage
-----
  python main.py                    # full pipeline
  python main.py --mode data        # input data only
  python main.py --mode plot        # figures only
  python main.py --seed 123         # custom random seed
  python main.py --pop 50 --gen 40  # quick test run

Pipeline stages
---------------
  Stage 0 – Setup: directories, logging, seed
  Stage 1 – Input data generation (CSV files)
  Stage 2 – GAN training and scenario generation
  Stage 3 – Surrogate NN training
  Stage 4 – NSGA-II multi-objective optimization
  Stage 5 – Decision making (Knee-point + TOPSIS)
  Stage 6 – Full simulation of selected solution
  Stage 7 – Baseline (no-ESS) evaluation
  Stage 8 – Result export (CSV + JSON)
  Stage 9 – Figure generation (18 plots)
  Stage 10– Summary table
"""

import os
import sys
import argparse
import json
import numpy as np
from datetime import datetime

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Project imports ──────────────────────────────────────────────────────────
from ship_ess import config as cfg
from ship_ess.dataio.utils import (get_logger, ensure_output_dirs, print_banner,
                              print_table, Timer, set_seed, save_json,
                              load_json, export_config)
from ship_ess.dataio.data_generator import (generate_all_inputs, load_all_profiles)
from ship_ess.learning.gan_scenarios import (train_gan, generate_scenarios, decode_scenarios,
                              compute_gan_metrics, save_gan_metrics)
from ship_ess.learning.surrogate import (train_surrogate, SurrogateNN)
from ship_ess.optimization.nsga2 import (run_nsga2, decode, save_pareto_csv,
                              save_history_csv, evaluate_population)
from ship_ess.optimization.decision_making import (select_solution)
from ship_ess.model.resilience import (simulate_ess, resilience_index,
                              cvar_resilience, evaluate_baseline,
                              make_gen_availability)
from ship_ess.viz.visualizer import generate_all_figures

# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ship ESS Resilience Optimization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode",  default="all",
                   choices=["all", "data", "gan", "surrogate",
                             "optimize", "plot", "experiments", "paper"],
                   help="Pipeline stage to run (default: all)")
    p.add_argument("--seed",  type=int, default=cfg.SEED,
                   help="Random seed (default: 42)")
    p.add_argument("--pop",   type=int, default=cfg.NSGA_POP_SIZE,
                   help="NSGA-II population size")
    p.add_argument("--gen",   type=int, default=cfg.NSGA_N_GEN,
                   help="NSGA-II number of generations")
    p.add_argument("--scenarios", type=int, default=cfg.N_SCENARIOS,
                   help="Number of GAN scenarios")
    p.add_argument("--gan-epochs", type=int, default=cfg.GAN_N_EPOCHS,
                   help="GAN training epochs")
    p.add_argument("--nn-samples", type=int, default=cfg.NN_TRAIN_SAMPLES,
                   help="Surrogate NN training samples")
    p.add_argument("--no-cv", action="store_true",
                   help="Skip 5-fold cross-validation")
    p.add_argument("--verbose", action="store_true", default=True,
                   help="Verbose output")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

def stage_setup(args: argparse.Namespace) -> None:
    """Stage 0: Create output directories, configure seed, export config."""
    ensure_output_dirs()
    set_seed(args.seed)
    export_config(os.path.join(cfg.RESULTS_DIR, "run_config.json"))


def stage_data(verbose: bool = True) -> dict:
    """
    Stage 1: Generate all input CSV data files.

    Returns
    -------
    dict with 'profiles' key containing loaded numpy arrays
    """
    print_banner("Stage 1 – Input Data Generation")
    with Timer("Data generation") as t:
        generate_all_inputs(verbose=verbose)
        profiles = load_all_profiles()
    return {"profiles": profiles}


def stage_gan(profiles: dict,
              n_epochs: int   = cfg.GAN_N_EPOCHS,
              n_scenarios: int = cfg.N_SCENARIOS,
              verbose: bool   = True) -> dict:
    """
    Stage 2: Train cGAN and generate uncertainty scenarios.

    Returns
    -------
    dict with 'gan', 'scenarios', 'g_losses', 'd_losses'
    """
    print_banner("Stage 2 – GAN Training & Scenario Generation")

    with Timer("GAN training") as t:
        gan, g_losses, d_losses = train_gan(
            profiles, n_epochs=n_epochs, verbose=verbose
        )

    with Timer("Scenario generation"):
        raw       = generate_scenarios(gan, n_samples=n_scenarios)
        scenarios = decode_scenarios(raw)
        # Phase 4.3: replace the weak cGAN wind with marine Engine-A wind when
        # SCENARIO_SOURCE=marine (+ EA_GENERATOR_CKPT set). pv/load unchanged.
        if os.environ.get("SCENARIO_SOURCE") == "marine":
            from ship_ess.learning.scenario_bridge import (wind_from_ea_samples,
                                                           default_ea_sampler)
            n = next(iter(scenarios.values())).shape[0]
            scenarios["wind"] = wind_from_ea_samples(
                default_ea_sampler(n, cfg.SEED),
                cfg.RES_RATED_KW[cfg.WIND_BUS]).astype("float32")

    gan_metrics = compute_gan_metrics(g_losses, d_losses, scenarios)
    save_gan_metrics(gan_metrics)

    # Attach deterministic generator availability AFTER GAN metrics, so the
    # metrics describe only GAN-generated channels; downstream stages
    # (surrogate labeling, CVaR) consume gen_avail from the returned dict.
    S = next(iter(scenarios.values())).shape[0]
    scenarios["gen_avail"] = make_gen_availability(S)

    return {
        "gan":       gan,
        "scenarios": scenarios,
        "g_losses":  g_losses,
        "d_losses":  d_losses,
        "gan_metrics": gan_metrics,
    }


def stage_surrogate(scenarios: dict,
                    n_samples: int  = cfg.NN_TRAIN_SAMPLES,
                    run_cv:    bool = True,
                    verbose:   bool = True) -> dict:
    """
    Stage 3: Generate training data and train the surrogate NN.

    Returns
    -------
    dict with 'nn', 'nn_losses', 'surrogate_metrics'
    """
    print_banner("Stage 3 – Surrogate Neural Network Training")

    with Timer("Surrogate training") as t:
        nn, metrics = train_surrogate(
            scenarios, n_samples=n_samples,
            run_cv=run_cv, verbose=verbose,
        )

    return {
        "nn":                nn,
        "nn_losses":         metrics["train_losses"],
        "surrogate_metrics": metrics,
    }


def stage_optimize(nn, pop_size: int = cfg.NSGA_POP_SIZE,
                   n_gen: int = cfg.NSGA_N_GEN,
                   verbose: bool = True) -> dict:
    """
    Stage 4: Run NSGA-II and save Pareto-front results.

    Returns
    -------
    dict with 'population', 'objectives', 'pareto_idx', 'history'
    """
    print_banner("Stage 4 – NSGA-II Multi-Objective Optimization")

    with Timer("NSGA-II") as t:
        result = run_nsga2(nn, pop_size=pop_size, n_gen=n_gen,
                            verbose=verbose)

    pop        = result["population"]
    obj        = result["objectives"]
    pareto_idx = result["pareto_front"]
    history    = result["history"]

    # Save results
    save_pareto_csv(pop, obj, pareto_idx)
    save_history_csv(history)

    return {
        "population": pop,
        "objectives": obj,
        "pareto_idx": pareto_idx,
        "history":    history,
    }


def stage_decision(pop: np.ndarray, obj: np.ndarray,
                   pareto_idx: list,
                   scenarios: dict,
                   verbose: bool = True) -> dict:
    """
    Stage 5: Apply Knee-point and TOPSIS decision making.
    Stage 6: Full simulation of the selected (knee-point) solution.
    Stage 7: Baseline (no-ESS) evaluation.

    Returns
    -------
    dict with full decision results + simulation outputs
    """
    print_banner("Stage 5 – Decision Making")
    from ship_ess.optimization.decision_making import find_knee_point, topsis

    pf_obj      = obj[pareto_idx]
    knee_idx    = find_knee_point(pf_obj)
    topsis_idx  = topsis(pf_obj)

    # Decode the knee-point solution
    best_chrom         = pop[pareto_idx[knee_idx]]
    ess_buses, E_kWh, P_kW = decode(best_chrom)

    # ── Stage 6: Full simulation ─────────────────────────────────────────
    print_banner("Stage 6 – Full Simulation of Selected Solution")
    mean_scenario = {k: v.mean(axis=0) for k, v in scenarios.items()}

    with Timer("Full simulation (mean scenario)"):
        sim_optimal = simulate_ess(ess_buses, E_kWh, P_kW, mean_scenario)

    with Timer("CVaR computation (all scenarios)"):
        cvar_opt, R_vals_opt = cvar_resilience(
            ess_buses, E_kWh, P_kW, scenarios)

    R_mean = float(resilience_index(sim_optimal))

    # Save decision results
    dec_result = select_solution(pop, obj, pareto_idx, cvar_opt=cvar_opt)

    # ── Stage 7: Baseline ────────────────────────────────────────────────
    print_banner("Stage 7 – Baseline (No-ESS) Evaluation")
    with Timer("Baseline CVaR computation"):
        cvar_no_ess, R_vals_no = evaluate_baseline(scenarios)

    # No-ESS trace under the mean scenario (for the resilience-trapezoid
    # schematic): a dummy ~0 ESS reuses the simulation engine, mirroring
    # evaluate_baseline().
    dummy_bus = [list(cfg.BASE_LOAD_KW.keys())[0]]
    sim_baseline = simulate_ess(
        dummy_bus,
        np.array([0.001], dtype=np.float32),
        np.array([0.001], dtype=np.float32),
        mean_scenario,
    )

    return {
        "ess_buses":   ess_buses,
        "E_kWh":       E_kWh,
        "P_kW":        P_kW,
        "knee_idx":    knee_idx,
        "topsis_idx":  topsis_idx,
        "sim_optimal": sim_optimal,
        "sim_baseline": sim_baseline,
        "cvar_opt":    cvar_opt,
        "R_vals_opt":  R_vals_opt,
        "R_mean":      R_mean,
        "cvar_no_ess": cvar_no_ess,
        "R_vals_no":   R_vals_no,
        "dec_result":  dec_result,
    }


def stage_export(all_results: dict) -> None:
    """
    Stage 8: Save scenario resilience CSV and summary JSON.
    """
    print_banner("Stage 8 – Result Export")

    # Scenario resilience CSV
    import pandas as pd
    from ship_ess.config import SCENARIO_RESIL_CSV, SUMMARY_JSON
    from ship_ess.dataio.utils import save_csv

    R_vals  = all_results["R_vals_opt"]
    R_df    = pd.DataFrame({
        "scenario_id":     np.arange(len(R_vals)),
        "resilience":      R_vals,
        "in_cvar_tail":    R_vals <= np.percentile(R_vals,
                                                    (1-cfg.ALPHA_CVAR)*100),
    })
    save_csv(R_df, SCENARIO_RESIL_CSV)

    # Summary JSON
    summary = {
        "timestamp":          datetime.now().isoformat(),
        "seed":               int(cfg.SEED),
        "pareto_front_size":  len(all_results["pareto_idx"]),
        "n_ess":              len(all_results["ess_buses"]),
        "ess_buses":          all_results["ess_buses"],
        "E_total_kWh":        round(float(all_results["E_kWh"].sum()), 2),
        "P_total_kW":         round(float(all_results["P_kW"].sum()), 2),
        "cost_usd":           round(float(
            all_results["objectives"][
                all_results["pareto_idx"][all_results["knee_idx"]], 0]), 2),
        "weight_kg":          round(float(
            all_results["objectives"][
                all_results["pareto_idx"][all_results["knee_idx"]], 1]), 2),
        "cvar_with_ess":      round(float(all_results["cvar_opt"]), 5),
        "cvar_no_ess":        round(float(all_results["cvar_no_ess"]), 5),
        "resilience_improvement_pp": round(
            (all_results["cvar_opt"] - all_results["cvar_no_ess"]) * 100, 3),
        "R_mean_fault":       round(float(all_results["R_mean"]), 5),
        "surrogate_test_mae": round(
            float(all_results["surrogate_metrics"]["test_mae"]), 6),
        "surrogate_test_r2":  round(
            float(all_results["surrogate_metrics"]["test_r2"]), 4),
    }
    save_json(summary, SUMMARY_JSON)

    return summary


def stage_figures(all_results: dict) -> dict:
    """Stage 9: Generate all 18 figures."""
    print_banner("Stage 9 – Figure Generation")
    return generate_all_figures(all_results)


def stage_summary(summary: dict, timings: dict) -> None:
    """Stage 10: Print final summary table to console."""
    print_banner("Final Results Summary")
    rows = [
        ("Pareto front size",            summary["pareto_front_size"]),
        ("ESS units installed",          summary["n_ess"]),
        ("ESS buses",                    str(summary["ess_buses"])),
        ("Total energy capacity (kWh)",  f"{summary['E_total_kWh']:.1f}"),
        ("Total power rating (kW)",      f"{summary['P_total_kW']:.1f}"),
        ("Installation cost ($)",        f"${summary['cost_usd']:,.0f}"),
        ("Added weight (kg)",            f"{summary['weight_kg']:.1f}"),
        ("CVaR₀.₉₅ resilience (w/ ESS)", f"{summary['cvar_with_ess']:.5f}"),
        ("CVaR₀.₉₅ resilience (no ESS)", f"{summary['cvar_no_ess']:.5f}"),
        ("Resilience improvement (pp)",  f"+{summary['resilience_improvement_pp']:.3f}"),
        ("Surrogate test MAE",           f"{summary['surrogate_test_mae']:.6f}"),
        ("Surrogate test R²",            f"{summary['surrogate_test_r2']:.4f}"),
        ("─" * 30,                       "─" * 12),
    ]
    for stage, elapsed in timings.items():
        rows.append((f"  Runtime – {stage}", f"{elapsed:.1f}s"))

    print_table(rows, header=("Metric", "Value"))
    print(f"\n  Output directory: {cfg.DATA_OUTPUT_DIR}")
    print(f"  Figures:          {cfg.FIGURES_DIR}")
    print(f"  Results:          {cfg.RESULTS_DIR}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    logger  = get_logger("main",
                          os.path.join(cfg.RESULTS_DIR, "pipeline.log"))
    timings = {}

    print_banner("Ship ESS Resilience Optimization – Pipeline Start")
    logger.info(f"Mode: {args.mode} | Seed: {args.seed} | "
                f"Pop: {args.pop} | Gen: {args.gen}")

    # Stage 0: Setup
    stage_setup(args)

    # ── Phase-3 statistical studies ──────────────────────────────────────
    if args.mode == "experiments":
        from ship_ess.run_experiments import main as run_exp_main
        sys.argv = [sys.argv[0]]              # run all studies
        run_exp_main()
        return

    # ── Phase-5 paper assets ─────────────────────────────────────────────
    if args.mode == "paper":
        from ship_ess.paper.build import build_all
        build_all()
        return

    # ── Data only ────────────────────────────────────────────────────────
    if args.mode == "data":
        with Timer("Data") as t:
            stage_data(verbose=args.verbose)
        timings["data"] = t.elapsed
        stage_summary({"pareto_front_size": 0, "n_ess": 0, "ess_buses": [],
                        "E_total_kWh": 0, "P_total_kW": 0, "cost_usd": 0,
                        "weight_kg": 0, "cvar_with_ess": 0, "cvar_no_ess": 0,
                        "resilience_improvement_pp": 0,
                        "surrogate_test_mae": 0, "surrogate_test_r2": 0},
                       timings)
        return

    # ── Full pipeline (default) ──────────────────────────────────────────
    all_results = {}

    # Stage 1
    with Timer("Data generation") as t:
        s1 = stage_data(verbose=args.verbose)
    timings["data"] = t.elapsed
    all_results.update(s1)
    all_results["hist_profiles"] = s1["profiles"]

    # Stage 2
    with Timer("GAN") as t:
        s2 = stage_gan(
            s1["profiles"],
            n_epochs=args.gan_epochs,
            n_scenarios=args.scenarios,
            verbose=args.verbose,
        )
    timings["GAN"] = t.elapsed
    all_results.update(s2)

    if args.mode == "gan":
        logger.info("Mode=gan: stopping after scenario generation.")
        return

    # Stage 3
    with Timer("Surrogate NN") as t:
        s3 = stage_surrogate(
            s2["scenarios"],
            n_samples=args.nn_samples,
            run_cv=not args.no_cv,
            verbose=args.verbose,
        )
    timings["surrogate"] = t.elapsed
    all_results.update(s3)

    if args.mode == "surrogate":
        logger.info("Mode=surrogate: stopping after NN training.")
        return

    # Stage 4
    with Timer("NSGA-II") as t:
        s4 = stage_optimize(
            s3["nn"],
            pop_size=args.pop,
            n_gen=args.gen,
            verbose=args.verbose,
        )
    timings["NSGA-II"] = t.elapsed
    all_results.update(s4)

    # Stages 5–7
    with Timer("Decision + Simulation") as t:
        s5 = stage_decision(
            s4["population"], s4["objectives"], s4["pareto_idx"],
            s2["scenarios"], verbose=args.verbose,
        )
    timings["decision+sim"] = t.elapsed
    all_results.update(s5)

    # Stage 8
    summary = stage_export(all_results)
    timings["total"] = sum(timings.values())

    # Stage 9
    with Timer("Figures") as t:
        paths = stage_figures(all_results)
    timings["figures"] = t.elapsed

    # Stage 10
    stage_summary(summary, timings)


if __name__ == "__main__":
    main()
