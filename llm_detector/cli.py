"""Command-line interface for the LLM Detection Pipeline."""

import os
import argparse
from collections import Counter, defaultdict

import sys

import pandas as pd

from llm_detector.pipeline import analyze_prompt
from llm_detector.calibration import (
    calibrate_from_baselines, save_calibration, load_calibration,
)
from llm_detector.baselines import analyze_baselines, collect_baselines
from llm_detector.similarity import analyze_similarity, print_similarity_report
from llm_detector.io import load_xlsx, load_csv, load_pdf


def _is_frozen():
    """Check if running as a PyInstaller bundle."""
    return getattr(sys, 'frozen', False)


def print_result(r, verbose=False):
    """Pretty-print a single result."""
    icons = {'RED': '\U0001f534', 'AMBER': '\U0001f7e0', 'YELLOW': '\U0001f7e1',
             'GREEN': '\U0001f7e2', 'MIXED': '\U0001f535', 'REVIEW': '\u26aa'}
    icon = icons.get(r['determination'], '?')

    print(f"\n  {icon} [{r['determination']}] {r['task_id'][:20]}  |  {r['occupation'][:45]}")
    print(f"     Attempter: {r['attempter'] or '(unknown)'} | Stage: {r['stage']} | Words: {r['word_count']} | Mode: {r.get('mode', '?')}")
    print(f"     Reason: {r['reason']}")

    cal_conf = r.get('calibrated_confidence')
    p_val = r.get('p_value')
    if cal_conf is not None and cal_conf != r.get('confidence'):
        cal_str = f"     Calibrated: conf={cal_conf:.3f}"
        if p_val is not None:
            cal_str += f"  p={p_val:.3f}"
        cal_str += f"  [{r.get('calibration_stratum', '?')}]"
        print(cal_str)

    if verbose or r['determination'] in ('RED', 'AMBER'):
        delta = r.get('norm_obfuscation_delta', 0)
        lang = r.get('lang_support_level', 'SUPPORTED')
        if delta > 0 or lang != 'SUPPORTED':
            print(f"     NORM obfuscation: {delta:.1%}  invisible={r.get('norm_invisible_chars', 0)} homoglyphs={r.get('norm_homoglyphs', 0)}")
            print(f"     GATE support:     {lang} (fw_coverage={r.get('lang_fw_coverage', 0):.2f}, non_latin={r.get('lang_non_latin_ratio', 0):.2f})")
        print(f"     Preamble:         {r['preamble_score']:.2f} ({r['preamble_severity']}, {r['preamble_hits']} hits)")
        if r['preamble_details']:
            for name, sev in r['preamble_details']:
                print(f"         -> [{sev}] {name}")
        print(f"     Fingerprints:     {r['fingerprint_score']:.2f} ({r['fingerprint_hits']} hits)")
        print(f"     Prompt Sig:       {r['prompt_signature_composite']:.2f}")
        print(f"         CFD={r['prompt_signature_cfd']:.3f} frames={r['prompt_signature_distinct_frames']} MFSR={r['prompt_signature_mfsr']:.3f}")
        print(f"         meta={r['prompt_signature_meta_design']} FC={r['prompt_signature_framing']}/3 must={r['prompt_signature_must_rate']:.3f}/sent")
        print(f"         contractions={r['prompt_signature_contractions']} numbered_criteria={r['prompt_signature_numbered_criteria']}")
        print(f"     IDI:              {r['instruction_density_idi']:.1f}  (imp={r['instruction_density_imperatives']} cond={r['instruction_density_conditionals']} Y/N={r['instruction_density_binary_specs']} MISS={r['instruction_density_missing_refs']} flag={r['instruction_density_flag_count']})")
        print(f"     VSD:              {r['voice_dissonance_vsd']:.1f}  (voice={r['voice_dissonance_voice_score']:.1f} x spec={r['voice_dissonance_spec_score']:.1f})")
        print(f"         gated={'YES' if r['voice_dissonance_voice_gated'] else 'no'} casual={r['voice_dissonance_casual_markers']} typos={r['voice_dissonance_misspellings']}")
        print(f"         cols={r['voice_dissonance_camel_cols']} calcs={r['voice_dissonance_calcs']} hedges={r['voice_dissonance_hedges']}")
        if r.get('ssi_triggered'):
            print(f"     SSI:  TRIGGERED  (spec={r['voice_dissonance_spec_score']:.1f}, voice=0, hedges=0, {r['word_count']}w)")
        nssi_score = r.get('self_similarity_nssi_score', 0.0)
        nssi_signals = r.get('self_similarity_nssi_signals', 0)
        nssi_det = r.get('self_similarity_determination')
        if nssi_score > 0 or nssi_det:
            det_str = nssi_det or 'n/a'
            print(f"     NSSI:             {nssi_score:.3f}  ({nssi_signals} signals, det={det_str})")
            print(f"         formulaic={r.get('self_similarity_formulaic_density', 0):.3f} power_adj={r.get('self_similarity_power_adj_density', 0):.3f}"
                  f" demo={r.get('self_similarity_demonstrative_density', 0):.3f} trans={r.get('self_similarity_transition_density', 0):.3f}")
            print(f"         sent_cv={r.get('self_similarity_sent_length_cv', 0):.3f} comp_ratio={r.get('self_similarity_comp_ratio', 0):.3f}"
                  f" hapax={r.get('self_similarity_hapax_ratio', 0):.3f} (unique={r.get('self_similarity_unique_words', 0)})")
        bscore = r.get('continuation_bscore', 0.0)
        dna_det = r.get('continuation_determination')
        if bscore > 0 or dna_det:
            det_str = dna_det or 'n/a'
            print(f"     DNA-GPT:          BScore={bscore:.4f}  (max={r.get('continuation_bscore_max', 0):.4f}, "
                  f"samples={r.get('continuation_n_samples', 0)}, det={det_str})")

        cd = r.get('channel_details', {})
        if cd.get('channels'):
            print(f"     -- Channels --")
            for ch_name, ch_info in cd['channels'].items():
                if ch_info['severity'] != 'GREEN':
                    eligible = 'Y' if ch_info.get('mode_eligible') else 'o'
                    print(f"     {eligible} {ch_name:18s} {ch_info['severity']:6s} score={ch_info['score']:.2f}  {ch_info['explanation'][:60]}")


def main():
    parser = argparse.ArgumentParser(description='LLM Detection Pipeline v0.61')
    parser.add_argument('input', nargs='?', help='Input file (.xlsx, .csv, or .pdf)')
    parser.add_argument('--gui', action='store_true', help='Launch desktop GUI mode')
    parser.add_argument('--text', help='Analyze a single text string')
    parser.add_argument('--sheet', help='Sheet name for xlsx files')
    parser.add_argument('--prompt-col', default='prompt', help='Column name containing prompts')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all layer details')
    parser.add_argument('--output', '-o', help='Output CSV path')
    parser.add_argument('--attempter', help='Filter by attempter name (substring match)')
    parser.add_argument('--no-similarity', action='store_true',
                        help='Skip cross-submission similarity analysis')
    parser.add_argument('--similarity-threshold', type=float, default=0.40,
                        help='Jaccard threshold for text similarity (default: 0.40)')
    parser.add_argument('--collect', metavar='PATH',
                        help='Append scored results to JSONL file for baseline accumulation')
    parser.add_argument('--analyze-baselines', metavar='JSONL',
                        help='Read accumulated JSONL and print per-occupation percentile tables')
    parser.add_argument('--baselines-csv', metavar='PATH',
                        help='Write baseline percentile tables to CSV (use with --analyze-baselines)')
    parser.add_argument('--no-layer3', action='store_true',
                        help='Skip Layer 3 entirely (NSSI + DNA-GPT)')
    parser.add_argument('--api-key', metavar='KEY',
                        help='API key for DNA-GPT continuation analysis. Falls back to '
                             'ANTHROPIC_API_KEY or OPENAI_API_KEY env var.')
    parser.add_argument('--provider', default='anthropic', choices=['anthropic', 'openai'],
                        help='LLM provider for DNA-GPT (default: anthropic)')
    parser.add_argument('--dna-model', metavar='MODEL',
                        help='Model name for DNA-GPT (default: auto per provider)')
    parser.add_argument('--dna-samples', type=int, default=3,
                        help='Number of regeneration samples for DNA-GPT (default: 3)')
    parser.add_argument('--mode', default='auto', choices=['task_prompt', 'generic_aigt', 'auto'],
                        help='Detection mode: task_prompt (prompt-structure primary), '
                             'generic_aigt (all channels), auto (heuristic). Default: auto')
    parser.add_argument('--calibrate', metavar='JSONL',
                        help='Build calibration table from labeled baseline JSONL and save to --cal-table')
    parser.add_argument('--cal-table', metavar='JSON',
                        help='Path to calibration table JSON (load for scoring, or save target for --calibrate)')
    args = parser.parse_args()

    if args.gui:
        from llm_detector.gui import launch_gui
        launch_gui()
        return

    if not args.api_key:
        env_key = 'ANTHROPIC_API_KEY' if args.provider == 'anthropic' else 'OPENAI_API_KEY'
        args.api_key = os.environ.get(env_key)

    if args.analyze_baselines:
        if not os.path.exists(args.analyze_baselines):
            print(f"ERROR: File not found: {args.analyze_baselines}")
            return
        analyze_baselines(args.analyze_baselines, output_csv=args.baselines_csv)
        return

    if args.calibrate:
        if not os.path.exists(args.calibrate):
            print(f"ERROR: File not found: {args.calibrate}")
            return
        cal = calibrate_from_baselines(args.calibrate)
        if cal is None:
            print("ERROR: Insufficient labeled human data for calibration (need >=20)")
            return
        cal_path = args.cal_table or args.calibrate.replace('.jsonl', '_calibration.json')
        save_calibration(cal, cal_path)
        print(f"  Global quantiles: {cal['global']}")
        print(f"  Strata: {len(cal.get('strata', {}))} domain x length_bin tables")
        return

    cal_table = None
    if args.cal_table and os.path.exists(args.cal_table):
        cal_table = load_calibration(args.cal_table)
        print(f"Loaded calibration table: {cal_table['n_calibration']} records, "
              f"{len(cal_table.get('strata', {}))} strata")

    run_l3 = not args.no_layer3

    if args.text:
        result = analyze_prompt(
            args.text, run_l3=run_l3,
            api_key=args.api_key, dna_provider=args.provider,
            dna_model=args.dna_model, dna_samples=args.dna_samples,
            mode=args.mode, cal_table=cal_table,
        )
        print_result(result, verbose=True)
        return

    if not args.input:
        if _is_frozen():
            from llm_detector.gui import launch_gui
            launch_gui()
            return
        parser.print_help()
        return

    ext = os.path.splitext(args.input)[1].lower()
    if ext in ('.xlsx', '.xlsm'):
        tasks = load_xlsx(args.input, sheet=args.sheet, prompt_col=args.prompt_col)
    elif ext == '.csv':
        tasks = load_csv(args.input, prompt_col=args.prompt_col)
    elif ext == '.pdf':
        tasks = load_pdf(args.input)
    else:
        print(f"ERROR: Unsupported file type: {ext}")
        return

    if not tasks:
        print("ERROR: No tasks found.")
        return

    if args.attempter:
        tasks = [t for t in tasks if args.attempter.lower() in t.get('attempter', '').lower()]
        print(f"Filtered to {len(tasks)} tasks matching attempter '{args.attempter}'")

    layer3_label = " + L3" if run_l3 else ""
    dna_label = " + DNA-GPT" if args.api_key else ""
    print(f"Processing {len(tasks)} tasks through pipeline v0.61{layer3_label}{dna_label}...")

    results = []
    text_map = {}
    for i, task in enumerate(tasks):
        r = analyze_prompt(
            task['prompt'],
            task_id=task.get('task_id', ''),
            occupation=task.get('occupation', ''),
            attempter=task.get('attempter', ''),
            stage=task.get('stage', ''),
            run_l3=run_l3,
            api_key=args.api_key,
            dna_provider=args.provider,
            dna_model=args.dna_model,
            dna_samples=args.dna_samples,
            mode=args.mode,
            cal_table=cal_table,
        )
        results.append(r)
        tid = task.get('task_id', f'_row{i}')
        text_map[tid] = task['prompt']
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(tasks)}...")

    det_counts = Counter(r['determination'] for r in results)
    print(f"\n{'='*90}")
    print(f"  PIPELINE v0.61 RESULTS (n={len(results)})")
    print(f"{'='*90}")
    for det in ['RED', 'AMBER', 'YELLOW', 'GREEN']:
        ct = det_counts.get(det, 0)
        pct = ct / len(results) * 100
        icons = {'RED': '\U0001f534', 'AMBER': '\U0001f7e0', 'YELLOW': '\U0001f7e1', 'GREEN': '\U0001f7e2'}
        print(f"  {icons[det]} {det:>8}: {ct:>4} ({pct:.1f}%)")

    flagged = [r for r in results if r['determination'] in ('RED', 'AMBER')]
    if flagged:
        print(f"\n{'='*90}")
        print(f"  FLAGGED SUBMISSIONS: {len(flagged)}")
        print(f"{'='*90}")
        for r in sorted(flagged, key=lambda x: x['confidence'], reverse=True):
            print_result(r, verbose=args.verbose)

    yellow = [r for r in results if r['determination'] == 'YELLOW']
    if yellow:
        print(f"\n  YELLOW ({len(yellow)} minor signals):")
        for r in sorted(yellow, key=lambda x: x['confidence'], reverse=True)[:10]:
            print(f"    \U0001f7e1 {r['task_id'][:12]:12} {r['occupation'][:40]:40} | {r['reason'][:50]}")

    if not args.no_similarity and len(results) >= 2:
        sim_pairs = analyze_similarity(
            results, text_map,
            jaccard_threshold=args.similarity_threshold,
        )
        print_similarity_report(sim_pairs)
    else:
        sim_pairs = []

    default_name = os.path.basename(args.input).rsplit('.', 1)[0] + '_pipeline_v061.csv'
    input_dir = os.path.dirname(os.path.abspath(args.input))
    output_path = args.output or os.path.join(input_dir, default_name)

    flat = []
    for r in results:
        row = {k: v for k, v in r.items() if k != 'preamble_details'}
        row['preamble_details'] = str(r.get('preamble_details', []))
        flat.append(row)

    if sim_pairs:
        sim_lookup = defaultdict(list)
        for p in sim_pairs:
            sim_lookup[p['id_a']].append(f"{p['id_b']}(J={p['jaccard']:.2f})")
            sim_lookup[p['id_b']].append(f"{p['id_a']}(J={p['jaccard']:.2f})")
        for row in flat:
            tid = row.get('task_id', '')
            row['similarity_flags'] = '; '.join(sim_lookup.get(tid, []))

    pd.DataFrame(flat).to_csv(output_path, index=False)
    print(f"\n  Results saved to: {output_path}")

    if args.collect:
        collect_baselines(results, args.collect)


def main_gui():
    """Entry point that always launches the GUI (for gui-scripts / executable)."""
    from llm_detector.gui import launch_gui
    launch_gui()


if __name__ == '__main__':
    main()
