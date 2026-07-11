"""Command-line entry point: ``vocdenoiser <group> <command> ...``.

Groups:
  snr     call-agnostic SNR scoring + clean-subset selection
          (scan | report | select | validate)
  search  autoresearch-style architecture search
          (run | report)
"""

from __future__ import annotations

import argparse
import sys

from vocdenoiser.snr.metric import SNRParams


def _snr_params(args) -> SNRParams:
    return SNRParams(n_fft=args.n_fft, hop=args.hop, active_db=args.active_db)


def _add_stft_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--active-db", type=float, default=6.0)


def cmd_scan(args) -> None:
    from vocdenoiser.snr.scan import scan_folder

    scan_folder(args.folder, args.out, params=_snr_params(args),
                workers=args.workers, limit=args.limit)


def cmd_report(args) -> None:
    from vocdenoiser.snr.report import build_report

    build_report(args.csv, args.out_dir)


def cmd_select(args) -> None:
    from vocdenoiser.snr.select import select_clean

    select_clean(
        args.csv,
        args.src_dir,
        args.out,
        snr_threshold=args.threshold,
        keep_percentile=args.keep_percentile,
        exclude_multi_source=not args.keep_multi_source,
        link_dir=args.link_dir,
    )


def cmd_validate(args) -> None:
    from vocdenoiser.snr.validate import run_validation

    run_validation(
        args.csv,
        args.src_dir,
        noise_dirs=args.noise_dir,
        out_dir=args.out_dir,
        n_clean=args.n_clean,
        params=_snr_params(args),
    )


def cmd_search_run(args) -> None:
    from vocdenoiser.search.ledger import Ledger
    from vocdenoiser.search.loop import SearchConfig, run_search
    from vocdenoiser.search.propose import Proposer

    if args.harness == "mock":
        from vocdenoiser.search.harness import MockHarness

        harness = MockHarness()
    else:
        from vocdenoiser.search.harness import TorchHarness

        overrides = {}
        if args.data_root:
            overrides["data_root"] = args.data_root
        harness = TorchHarness(base_config_overrides=overrides, max_steps=args.max_steps)

    ledger = Ledger(args.ledger)
    proposer = Proposer(frontier_k=args.frontier_k)
    cfg = SearchConfig(
        iters=args.iters,
        seeds=tuple(args.seeds),
        k_sigma=args.k_sigma,
        frontier_k=args.frontier_k,
        seed=args.seed,
    )
    best = run_search(harness, ledger, proposer, cfg)
    if best:
        print(f"\nBest: metric={best.metric:+.3f} id={best.id}\n{best.candidate}")


def cmd_search_report(args) -> None:
    from vocdenoiser.search.report import build_report

    print(build_report(args.ledger))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vocdenoiser")
    sub = parser.add_subparsers(dest="group", required=True)

    snr = sub.add_parser("snr", help="SNR scoring + selection")
    snr_sub = snr.add_subparsers(dest="cmd", required=True)

    ps = snr_sub.add_parser("scan", help="score every WAV in a folder -> CSV")
    ps.add_argument("folder")
    ps.add_argument("--out", required=True, help="output CSV path")
    ps.add_argument("--workers", type=int, default=None)
    ps.add_argument("--limit", type=int, default=None)
    _add_stft_args(ps)
    ps.set_defaults(func=cmd_scan)

    pr = snr_sub.add_parser("report", help="distribution report from a scan CSV")
    pr.add_argument("csv")
    pr.add_argument("--out-dir", default="reports")
    pr.set_defaults(func=cmd_report)

    pl = snr_sub.add_parser("select", help="apply a threshold -> clean manifest")
    pl.add_argument("csv")
    pl.add_argument("--src-dir", required=True, help="folder the clips live in")
    pl.add_argument("--out", required=True, help="output manifest CSV")
    pl.add_argument("--threshold", type=float, default=None, help="absolute snr_db cutoff")
    pl.add_argument("--keep-percentile", type=float, default=None, help="keep top X%%")
    pl.add_argument("--keep-multi-source", action="store_true",
                    help="do NOT drop clips flagged with >1 active segment")
    pl.add_argument("--link-dir", default=None, help="symlink selected clips here")
    pl.set_defaults(func=cmd_select)

    pv = snr_sub.add_parser("validate", help="call-agnosticism validation")
    pv.add_argument("csv")
    pv.add_argument("--src-dir", required=True)
    pv.add_argument("--noise-dir", action="append", required=True,
                    help="noise folder (repeatable), e.g. --noise-dir data/Noise --noise-dir data/Cigarra")
    pv.add_argument("--out-dir", default="reports")
    pv.add_argument("--n-clean", type=int, default=120)
    _add_stft_args(pv)
    pv.set_defaults(func=cmd_validate)

    search = sub.add_parser("search", help="architecture search")
    search_sub = search.add_subparsers(dest="cmd", required=True)

    sr = search_sub.add_parser("run", help="run the search loop")
    sr.add_argument("--harness", choices=["mock", "torch"], default="mock",
                    help="mock = synthetic landscape (no GPU); torch = real training")
    sr.add_argument("--ledger", default="artifacts/search_ledger.jsonl")
    sr.add_argument("--iters", type=int, default=40)
    sr.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    sr.add_argument("--k-sigma", type=float, default=1.0)
    sr.add_argument("--frontier-k", type=int, default=8)
    sr.add_argument("--seed", type=int, default=0)
    sr.add_argument("--max-steps", type=int, default=400, help="torch harness compute budget")
    sr.add_argument("--data-root", default=None, help="clean-call root (torch harness)")
    sr.set_defaults(func=cmd_search_run)

    srep = search_sub.add_parser("report", help="summarise a search ledger")
    srep.add_argument("--ledger", default="artifacts/search_ledger.jsonl")
    srep.set_defaults(func=cmd_search_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
