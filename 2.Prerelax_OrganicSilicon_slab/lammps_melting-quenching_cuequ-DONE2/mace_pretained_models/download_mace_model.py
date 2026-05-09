#!/usr/bin/env python3
"""
Download MACE pretrained models for LAMMPS simulations.
Supports MACE-MP, MACE-OFF, MACE-ANI, and MACE-OMOL models.

Usage:
    python download_mace_model.py [model_type] [model_size] [--output-dir OUTPUT_DIR]

Examples:
    python download_mace_model.py mp medium-mpa-0
    python download_mace_model.py mp small --output-dir ./models
    python download_mace_model.py off medium
    python download_mace_model.py anicc
"""

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    from mace.calculators.foundations_models import (
        mace_mp_urls,
        mace_mp_names,
    )
except ImportError as e:
    print(f"Error: Failed to import MACE modules: {e}")
    print("Please ensure MACE is properly installed:")
    print("  pip install mace-torch")
    sys.exit(1)


def download_mace_mp_model(model_size="medium-mpa-0", output_dir=None):
    """
    Download MACE-MP model (Materials Project, 89 elements).
    
    Args:
        model_size: Model size specification
        output_dir: Optional output directory (default: current directory)
    
    Returns:
        Path to downloaded model file
    """
    print(f"Downloading MACE-MP model: {model_size}")
    print("This model supports 89 elements including C, F, O, H")
    print("License: MIT")
    print()
    
    try:
        # Get URL from MACE URL dictionary
        checkpoint_url = (
            mace_mp_urls.get(model_size, mace_mp_urls["medium-mpa-0"])
            if model_size in mace_mp_names
            else model_size
        )
        
        # Check for ASL license models
        ASL_checkpoint_urls = {
            mace_mp_urls.get("small-omat-0"),
            mace_mp_urls.get("medium-omat-0"),
            mace_mp_urls.get("mace-matpes-pbe-0"),
            mace_mp_urls.get("mace-matpes-r2scan-0"),
        }
        if checkpoint_url in ASL_checkpoint_urls:
            print(
                "Using model under Academic Software License (ASL) license, see https://github.com/gabor1/ASL"
            )
            print("To use this model you accept the terms of the license.")
            print()
        
        # Extract original filename from URL
        original_filename = os.path.basename(checkpoint_url)
        # Remove query parameters if present (e.g., ?raw=true)
        if "?" in original_filename:
            original_filename = original_filename.split("?")[0]
        
        # Determine output directory
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = Path(".")  # Current directory
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / original_filename
        
        # Check if file already exists
        if output_file.exists():
            print(f"Model file already exists: {output_file}")
            print("Skipping download.")
            return str(output_file)
        
        # Download directly to output directory
        print(f"Downloading from: {checkpoint_url}")
        print(f"Saving to: {output_file}")
        
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                if block_num % 10 == 0:  # Print every 10 blocks
                    print(f"\rDownloading: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)", end="", flush=True)
        
        try:
            urllib.request.urlretrieve(checkpoint_url, output_file, reporthook=report_progress)
            print("\nModel downloaded successfully!")
            print(f"Location: {output_file}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Model download failed (HTTP {e.code}): {checkpoint_url}. Please check the URL."
            ) from e
        except Exception as e:
            # Clean up partial download on error
            if output_file.exists():
                output_file.unlink()
            raise RuntimeError(f"Model download failed: {e}") from e
        
        return str(output_file)
    except Exception as e:
        print(f"Error downloading MACE-MP model: {e}")
        raise


def download_mace_off_model(model_size="medium", output_dir=None):
    """
    Download MACE-OFF23 model (organic molecules).
    
    Args:
        model_size: Model size ('small', 'medium', 'large')
        output_dir: Optional output directory (default: current directory)
    
    Returns:
        Path to downloaded model file
    """
    print(f"Downloading MACE-OFF23 model: {model_size}")
    print("This model is for organic molecules")
    print("License: ASL (Academic Software License - non-commercial use only)")
    print()
    
    try:
        # MACE-OFF23 URLs
        urls = {
            "small": "https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23_small.model",
            "medium": "https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23_medium.model",
            "large": "https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23_large.model",
        }
        
        checkpoint_url = urls.get(model_size, urls["medium"])
        original_filename = f"MACE-OFF23_{model_size}.model"
        
        # Determine output directory
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = Path(".")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / original_filename
        
        # Check if file already exists
        if output_file.exists():
            print(f"Model file already exists: {output_file}")
            print("Skipping download.")
            return str(output_file)
        
        # Download directly to output directory
        print(f"Downloading from: {checkpoint_url}")
        print(f"Saving to: {output_file}")
        
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                if block_num % 10 == 0:
                    print(f"\rDownloading: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)", end="", flush=True)
        
        try:
            urllib.request.urlretrieve(checkpoint_url, output_file, reporthook=report_progress)
            print("\nModel downloaded successfully!")
            print(f"Location: {output_file}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Model download failed (HTTP {e.code}): {checkpoint_url}. Please check the URL."
            ) from e
        except Exception as e:
            # Clean up partial download on error
            if output_file.exists():
                output_file.unlink()
            raise RuntimeError(f"Model download failed: {e}") from e
        
        return str(output_file)
    except Exception as e:
        print(f"Error downloading MACE-OFF model: {e}")
        raise


def download_mace_anicc_model(output_dir=None):
    """
    Download MACE-ANI model (H, C, N, O).
    
    Args:
        output_dir: Optional output directory (default: current directory)
    
    Returns:
        Path to downloaded model file
    """
    print("Downloading MACE-ANI model")
    print("This model supports H, C, N, O elements")
    print("License: MIT")
    print()
    
    try:
        # MACE-ANI model URL
        checkpoint_url = "https://github.com/ACEsuit/mace/raw/main/mace/calculators/foundations_models/ani500k_large_CC.model"
        original_filename = "ani500k_large_CC.model"
        
        # Determine output directory
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = Path(".")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / original_filename
        
        # Check if file already exists
        if output_file.exists():
            print(f"Model file already exists: {output_file}")
            print("Skipping download.")
            return str(output_file)
        
        # Download directly to output directory
        print(f"Downloading from: {checkpoint_url}")
        print(f"Saving to: {output_file}")
        
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                if block_num % 10 == 0:
                    print(f"\rDownloading: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)", end="", flush=True)
        
        try:
            urllib.request.urlretrieve(checkpoint_url, output_file, reporthook=report_progress)
            print("\nModel downloaded successfully!")
            print(f"Location: {output_file}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Model download failed (HTTP {e.code}): {checkpoint_url}. Please check the URL."
            ) from e
        except Exception as e:
            # Clean up partial download on error
            if output_file.exists():
                output_file.unlink()
            raise RuntimeError(f"Model download failed: {e}") from e
        
        return str(output_file)
    except Exception as e:
        print(f"Error downloading MACE-ANI model: {e}")
        raise


def download_mace_omol_model(output_dir=None):
    """
    Download MACE-OMOL model.
    
    Args:
        output_dir: Optional output directory (default: current directory)
    
    Returns:
        Path to downloaded model file
    """
    print("Downloading MACE-OMOL model")
    print("License: ASL (Academic Software License - non-commercial use only)")
    print()
    
    try:
        # MACE-OMOL model URL
        checkpoint_url = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/MACE-omol-0-extra-large-1024.model"
        original_filename = "MACE-omol-0-extra-large-1024.model"
        
        # Determine output directory
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = Path(".")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / original_filename
        
        # Check if file already exists
        if output_file.exists():
            print(f"Model file already exists: {output_file}")
            print("Skipping download.")
            return str(output_file)
        
        # Download directly to output directory
        print(f"Downloading from: {checkpoint_url}")
        print(f"Saving to: {output_file}")
        
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                if block_num % 10 == 0:
                    print(f"\rDownloading: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)", end="", flush=True)
        
        try:
            urllib.request.urlretrieve(checkpoint_url, output_file, reporthook=report_progress)
            print("\nModel downloaded successfully!")
            print(f"Location: {output_file}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Model download failed (HTTP {e.code}): {checkpoint_url}. Please check the URL."
            ) from e
        except Exception as e:
            # Clean up partial download on error
            if output_file.exists():
                output_file.unlink()
            raise RuntimeError(f"Model download failed: {e}") from e
        
        return str(output_file)
    except Exception as e:
        print(f"Error downloading MACE-OMOL model: {e}")
        raise


def list_available_models():
    """List all available MACE model types and sizes."""
    print("Available MACE Pretrained Models:")
    print("=" * 60)
    print()
    
    print("1. MACE-MP (Materials Project - 89 elements)")
    print("   Sizes: small, medium, large, small-0b, medium-0b,")
    print("          small-0b2, medium-0b2, large-0b2, medium-0b3,")
    print("          medium-mpa-0 (default), small-omat-0, medium-omat-0")
    print("   License: MIT")
    print("   Recommended for: PTFE-water interface (C, F, O, H)")
    print()
    
    print("2. MACE-OFF23 (Organic molecules)")
    print("   Sizes: small, medium, large")
    print("   License: ASL (non-commercial use only)")
    print()
    
    print("3. MACE-ANI (H, C, N, O)")
    print("   Single model available")
    print("   License: MIT")
    print()
    
    print("4. MACE-OMOL")
    print("   Single model available (extra_large)")
    print("   License: ASL (non-commercial use only)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download MACE pretrained models for LAMMPS simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default MACE-MP model (medium-mpa-0)
  python download_mace_model.py mp
  
  # Download specific MACE-MP model size
  python download_mace_model.py mp small
  
  # Download to specific directory
  python download_mace_model.py mp medium-mpa-0 --output-dir ./models
  
  # Download MACE-OFF model
  python download_mace_model.py off medium
  
  # Download MACE-ANI model
  python download_mace_model.py anicc
  
  # List available models
  python download_mace_model.py --list
        """
    )
    
    parser.add_argument(
        "model_type",
        nargs="?",
        choices=["mp", "off", "anicc", "omol"],
        help="Model type: mp (MACE-MP), off (MACE-OFF), anicc (MACE-ANI), omol (MACE-OMOL)",
    )
    
    parser.add_argument(
        "model_size",
        nargs="?",
        help="Model size (for mp: small/medium/large/medium-mpa-0, for off: small/medium/large)",
    )
    
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory for downloaded model (default: current directory)",
    )
    
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available models and exit",
    )
    
    args = parser.parse_args()
    
    # List available models
    if args.list:
        list_available_models()
        sys.exit(0)
    
    # Require model type if not listing
    if not args.model_type:
        print("Error: Model type is required")
        print("Use --list to see available models")
        print("Use --help for usage information")
        sys.exit(1)
    
    # Download model based on type
    try:
        if args.model_type == "mp":
            model_size = args.model_size or "medium-mpa-0"
            model_path = download_mace_mp_model(model_size, args.output_dir)
        elif args.model_type == "off":
            model_size = args.model_size or "medium"
            model_path = download_mace_off_model(model_size, args.output_dir)
        elif args.model_type == "anicc":
            model_path = download_mace_anicc_model(args.output_dir)
        elif args.model_type == "omol":
            model_path = download_mace_omol_model(args.output_dir)
        
        print()
        print("=" * 60)
        print("Download completed successfully!")
        print(f"Model file: {model_path}")
        print()
        print("Next steps:")
        print("1. Convert model to MLIAP format for LAMMPS:")
        print(f"   python ../convert_mace_to_mliap.py {model_path}")
        print("2. Update model path in lammps_relax.inp and lammps_production.inp")
        
    except KeyboardInterrupt:
        print("\nDownload cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

