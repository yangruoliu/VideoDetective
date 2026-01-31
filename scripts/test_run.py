#!/usr/bin/env python3
"""
VideoDetective - Single Video Test Script

This script demonstrates the VideoDetective pipeline for video question answering.
It processes a single video with a multiple-choice question and generates:
1. The predicted answer
2. Belief evolution visualization (heatmap)
3. Detailed debug information in JSON format

Usage Examples:
    # Basic usage with command line arguments
    python scripts/test_run.py \
        --video_path /path/to/video.mp4 \
        --question "What is the man doing?" \
        --options "A. Running, B. Walking, C. Sitting, D. Standing"
    
    # Using Video-MME benchmark format
    python scripts/test_run.py \
        --tsv_path /path/to/Video-MME.tsv \
        --video_dir /path/to/videos \
        --index 0
    
    # With custom output directory
    python scripts/test_run.py \
        --video_path /path/to/video.mp4 \
        --question "..." \
        --options "..." \
        --output_dir ./output/my_test

Environment Setup:
    Before running, make sure to:
    1. Copy .env.example to .env
    2. Fill in your API key and other settings in .env
    
    Required environment variables:
    - VIDEODETECTIVE_API_KEY: Your VLM API key
    
    Optional environment variables:
    - VIDEODETECTIVE_BASE_URL: API base URL
    - VIDEODETECTIVE_VLM_MODEL: VLM model name
    - HF_CACHE_DIR: HuggingFace cache directory
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VideoDetective - Single Video Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Direct video input
    python scripts/test_run.py \\
        --video_path video.mp4 \\
        --question "What is happening?" \\
        --options "A. Action1, B. Action2, C. Action3, D. Action4"
    
    # Video-MME benchmark format
    python scripts/test_run.py \\
        --tsv_path benchmark.tsv \\
        --video_dir ./videos \\
        --index 0
        """
    )
    
    # Video input options (either direct video or benchmark TSV)
    video_group = parser.add_argument_group("Video Input")
    video_group.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to video file (use this for direct video input)"
    )
    video_group.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question to ask about the video"
    )
    video_group.add_argument(
        "--options",
        type=str,
        default=None,
        help="Answer options (e.g., 'A. Option1, B. Option2, C. Option3, D. Option4')"
    )
    video_group.add_argument(
        "--ground_truth",
        type=str,
        default=None,
        help="Ground truth answer (e.g., 'A') for evaluation"
    )
    
    # Benchmark input options
    benchmark_group = parser.add_argument_group("Benchmark Input (Video-MME format)")
    benchmark_group.add_argument(
        "--tsv_path",
        type=str,
        default=None,
        help="Path to Video-MME format TSV file"
    )
    benchmark_group.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Directory containing video files"
    )
    benchmark_group.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of sample in TSV file (0-indexed)"
    )
    
    # Pipeline settings
    pipeline_group = parser.add_argument_group("Pipeline Settings")
    pipeline_group.add_argument(
        "--max_steps",
        type=int,
        default=10,
        help="Maximum Bayesian search iterations (default: 10)"
    )
    pipeline_group.add_argument(
        "--total_budget",
        type=int,
        default=32,
        help="Total frames for final answer generation (default: 32)"
    )
    
    # Output settings
    output_group = parser.add_argument_group("Output Settings")
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for results (default: output)"
    )
    output_group.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose logging (default: True)"
    )
    output_group.add_argument(
        "--no_viz",
        action="store_true",
        help="Disable visualization output"
    )
    
    return parser.parse_args()


def load_video_mme_sample(tsv_path: str, index: int, video_dir: str) -> dict:
    """
    Load a sample from Video-MME format TSV file.
    
    Expected TSV columns:
        - video: Video ID/filename (without extension)
        - question: The question text
        - candidates/options: Answer options
        - answer: Ground truth answer (letter)
    
    Args:
        tsv_path: Path to TSV file
        index: Sample index (0-indexed)
        video_dir: Directory containing videos
    
    Returns:
        Dictionary with video_id, video_path, query, gt, question, candidates
    """
    df = pd.read_csv(tsv_path, sep='\t')
    
    if index < 0 or index >= len(df):
        raise ValueError(f"Index {index} out of range. TSV has {len(df)} samples.")
    
    row = df.iloc[index]
    
    # Extract fields with fallbacks
    video_id = row.get('video', row.get('video_id', f'sample_{index}'))
    question = row.get('question', '')
    candidates = row.get('candidates', row.get('options', ''))
    answer = row.get('answer', '')
    
    # Build query string
    query = f"{question} Options: {candidates}"
    
    # Find video file
    video_path = None
    possible_extensions = ['.mp4', '.avi', '.mkv', '.webm', '.mov']
    
    for ext in possible_extensions:
        candidate_path = Path(video_dir) / f"{video_id}{ext}"
        if candidate_path.exists():
            video_path = str(candidate_path)
            break
    
    if video_path is None:
        # Try searching in subdirectories
        for ext in possible_extensions:
            matches = list(Path(video_dir).rglob(f"{video_id}{ext}"))
            if matches:
                video_path = str(matches[0])
                break
    
    return {
        "video_id": video_id,
        "video_path": video_path,
        "query": query,
        "gt": answer,
        "question": question,
        "candidates": candidates
    }


def plot_belief_history(debug_info: dict, output_path: str, title: str = ""):
    """
    Plot belief evolution history as a heatmap.
    
    Generates a two-panel visualization:
    1. Top: Heatmap showing belief evolution across search steps
    2. Bottom: Final belief distribution with marked observation points
    
    Args:
        debug_info: Debug info from VideoDetective.solve()
        output_path: Path to save the visualization
        title: Additional title text
    """
    history = debug_info.get("belief_history", {})
    mu_snapshots = history.get("mu_snapshots", [])
    observations = history.get("observations", [])
    
    if len(mu_snapshots) < 1:
        print("No belief history to plot")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Belief evolution heatmap
    ax1 = axes[0]
    mu_matrix = np.array([m for m in mu_snapshots if len(m) > 0])
    
    if len(mu_matrix) > 0:
        im = ax1.imshow(
            mu_matrix,
            aspect='auto',
            cmap='hot',
            interpolation='nearest'
        )
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Search Step')
        ax1.set_title(f'Belief (μ) Evolution Over Search Steps\n{title}')
        plt.colorbar(im, ax=ax1, label='Relevance')
        
        # Mark observation points
        for i, obs in enumerate(observations):
            if obs.get("t_obs") is not None:
                ax1.axvline(x=obs["t_obs"], color='cyan', linestyle='--', alpha=0.5)
    
    # Plot 2: Final belief distribution
    ax2 = axes[1]
    if len(mu_snapshots) > 0 and len(mu_snapshots[-1]) > 0:
        final_mu = mu_snapshots[-1]
        x = np.arange(len(final_mu))
        
        ax2.fill_between(x, 0, final_mu, alpha=0.3, color='blue')
        ax2.plot(x, final_mu, 'b-', linewidth=2, label='Final Belief (μ)')
        
        # Mark visited points
        visited = debug_info.get("visited_indices", [])
        for v in visited:
            if v < len(final_mu):
                ax2.axvline(x=v, color='red', linestyle='--', alpha=0.5)
        
        # Mark coverage anchors
        coverage = debug_info.get("coverage_indices", [])
        for c in coverage:
            if c < len(final_mu):
                ax2.scatter([c], [final_mu[c]], color='green', s=100, marker='^', 
                           zorder=5, label='Coverage' if c == coverage[0] else '')
        
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Belief Value')
        ax2.set_title('Final Belief Distribution')
        ax2.legend(loc='upper right')
        ax2.set_xlim(0, len(final_mu))
        ax2.set_ylim(0, max(final_mu) * 1.1 if max(final_mu) > 0 else 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {output_path}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check environment setup
    api_key = os.getenv("VIDEODETECTIVE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("QWEN_API_KEY")
    if not api_key:
        print("=" * 60)
        print("ERROR: API key not configured!")
        print()
        print("Please set up your environment:")
        print("  1. Copy .env.example to .env")
        print("  2. Fill in your API key in .env")
        print()
        print("Or set the VIDEODETECTIVE_API_KEY environment variable:")
        print("  export VIDEODETECTIVE_API_KEY=your_api_key_here")
        print("=" * 60)
        sys.exit(1)
    
    # Import VideoDetective after path setup
    from src.pipeline import VideoDetective
    
    # Determine input mode and load sample
    if args.video_path:
        # Direct video input mode
        if not args.question:
            print("ERROR: --question is required when using --video_path")
            sys.exit(1)
        if not args.options:
            print("ERROR: --options is required when using --video_path")
            sys.exit(1)
        
        if not Path(args.video_path).exists():
            print(f"ERROR: Video file not found: {args.video_path}")
            sys.exit(1)
        
        sample = {
            "video_id": Path(args.video_path).stem,
            "video_path": args.video_path,
            "query": f"{args.question} Options: {args.options}",
            "gt": args.ground_truth or "",
            "question": args.question,
            "candidates": args.options
        }
    elif args.tsv_path:
        # Video-MME benchmark mode
        if not args.video_dir:
            print("ERROR: --video_dir is required when using --tsv_path")
            sys.exit(1)
        
        try:
            sample = load_video_mme_sample(args.tsv_path, args.index, args.video_dir)
        except Exception as e:
            print(f"ERROR: Failed to load sample: {e}")
            sys.exit(1)
    else:
        print("ERROR: Please provide either --video_path or --tsv_path")
        print()
        print("Examples:")
        print("  # Direct video input:")
        print('  python scripts/test_run.py --video_path video.mp4 --question "What is happening?" --options "A. X, B. Y, C. Z, D. W"')
        print()
        print("  # Video-MME benchmark:")
        print("  python scripts/test_run.py --tsv_path benchmark.tsv --video_dir ./videos --index 0")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print run information
    print("=" * 60)
    print("VideoDetective Test Run")
    print("=" * 60)
    print(f"Video ID: {sample['video_id']}")
    print(f"Video Path: {sample['video_path']}")
    print(f"Question: {sample['question']}")
    print(f"Options: {sample['candidates']}")
    if sample['gt']:
        print(f"Ground Truth: {sample['gt']}")
    print("-" * 60)
    
    # Check video exists
    if sample['video_path'] is None or not Path(sample['video_path']).exists():
        print(f"ERROR: Video file not found: {sample['video_path']}")
        sys.exit(1)
    
    # Initialize VideoDetective
    print("Initializing VideoDetective...")
    detective = VideoDetective(verbose=args.verbose)
    
    # Run inference
    print("Running inference...")
    result = detective.solve(
        video_path=sample['video_path'],
        query=sample['query'],
        max_steps=args.max_steps,
        total_budget=args.total_budget
    )
    
    # Print results
    print()
    print("=" * 60)
    print("RESULTS:")
    print(f"  Prediction: {result.answer}")
    if sample['gt']:
        print(f"  Ground Truth: {sample['gt']}")
        
        # Simple accuracy check
        pred_letter = result.answer.strip().upper()[:1] if result.answer else ""
        gt_letter = sample['gt'].strip().upper()[:1] if sample['gt'] else ""
        is_correct = pred_letter == gt_letter
        
        print(f"  Correct: {'✓' if is_correct else '✗'}")
    print("=" * 60)
    
    # Generate visualization
    if not args.no_viz:
        video_id = sample['video_id']
        viz_path = output_dir / f"{video_id}_belief.png"
        plot_belief_history(
            result.debug_info,
            str(viz_path),
            title=f"Video: {video_id} | Q: {sample['question'][:50]}..."
        )
    
    # Save results
    video_id = sample['video_id']
    results_path = output_dir / f"{video_id}_results.json"
    
    results_data = {
        "video_id": video_id,
        "question": sample['question'],
        "candidates": sample['candidates'],
        "ground_truth": sample['gt'],
        "prediction": result.answer,
        "debug_info": result.debug_info
    }
    
    if sample['gt']:
        pred_letter = result.answer.strip().upper()[:1] if result.answer else ""
        gt_letter = sample['gt'].strip().upper()[:1] if sample['gt'] else ""
        results_data["is_correct"] = pred_letter == gt_letter
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, default=str, ensure_ascii=False)
    print(f"Saved results to: {results_path}")
    
    return result


if __name__ == "__main__":
    main()
