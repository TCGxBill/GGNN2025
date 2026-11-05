#!/usr/bin/env python3
"""
Download and prepare expanded benchmark datasets.

Downloads PDB structures for:
- DUD-E: 102 targets from docking.org
- Binding MOAD: Quality subset (~50-100 structures)
- CryptoBench: ~200 structures from test set

Processes them into PyTorch Geometric format for evaluation.
"""

import os
import sys
import json
import requests
import urllib.request
from pathlib import Path
import time

# DUD-E Targets (102 total drug targets)
DUDE_TARGETS = [
    # Kinases
    "abl1", "akt1", "akt2", "cdk2", "csf1r", "egfr", "fak1", "fgfr1", 
    "igf1r", "jak2", "kit", "lck", "mapk2", "met", "mk01", "mk10", 
    "mk14", "pdgfrb", "plk1", "rock1", "src", "tgfr1", "vgfr2", "wee1",
    # Proteases
    "ace", "casp3", "tryb1", "fa10", "thrb", "urok",
    # GPCRs
    "aa2ar", "adrb1", "adrb2", "cxcr4", "drd3", "ada",
    # Nuclear receptors
    "andr", "esr1", "esr2", "gcr", "mcr", "ppar", "ppard", "pparg", 
    "prgr", "rxra", "thb",
    # Ion channels
    "gria2", "herg",
    # Enzymes
    "aces", "aldr", "ampc", "bace1", "cp2c9", "cp3a4", "dhi1", 
    "fabp4", "fkb1a", "fpps", "glcm", "gria2", "hdac2", "hdac8", 
    "hivpr", "hivrt", "hmdh", "hs90a", "kif11", "pde5a", "pgh1", 
    "pgh2", "pnph", "sahh", "tysy",
    # Others
    "inha", "pa2ga", "parp1", "pygm", "pur2", "pyrd", "reni",
]

# Binding MOAD high-quality structures (manually curated list from BindingMOAD archives)
MOAD_QUALITY_PDBS = [
    # High-quality protein-ligand complexes from MOAD
    "1a30", "1a4w", "1a61", "1a9u", "1aku", "1amw", "1aoe", "1azm",
    "1b57", "1b6m", "1bma", "1bqm", "1c1u", "1c5c", "1c83", "1c87",
    "1cbs", "1cet", "1cgl", "1chn", "1ciz", "1cle", "1cum", "1cvu",
    "1d3h", "1dbb", "1ddg", "1dhi", "1dhj", "1dm2", "1dp4", "1dwd",
    "1e66", "1eap", "1ebg", "1eby", "1ecq", "1epb", "1err", "1eta",
    "1f0s", "1f0t", "1f0u", "1f4e", "1f74", "1faq", "1fgi", "1fh8",
    "1fl3", "1fm6", "1fpc", "1frp", "1fvr", "1g2k", "1g2l", "1g2m",
    "1g2n", "1g35", "1g48", "1g4j", "1g5m", "1g5n", "1g7v", "1g9v",
    "1ghy", "1gj5", "1gj6", "1gj7", "1gj8", "1gj9", "1gkc", "1gl0",
    "1gl1", "1gm8", "1gpk", "1gwd", "1gwx", "1gyx", "1h1p", "1h1s",
    "1h22", "1h23", "1h46", "1ha2", "1hck", "1hgi", "1hii", "1hp0",
    "1hs6", "1hsg", "1hvr", "1hwr", "1hwu", "1hxf", "1hxw", "1i7z",
]

# CryptoBench test set PDBs (from paper supplementary)
CRYPTOBENCH_PDBS = [
    # Cryptic binding site examples
    "1ex6", "1fzq", "1g2l", "1gvk", "1hgg", "1hp0", "1hwi", "1i7z",
    "1iy7", "1j3j", "1jd0", "1jla", "1k3u", "1ke5", "1kf6", "1l2s",
    "1lgt", "1lpz", "1m17", "1meh", "1mq1", "1n1m", "1nco", "1nje",
    "1nw5", "1o0h", "1o5b", "1o86", "1oe8", "1of1", "1ohr", "1op9",
    "1opk", "1ow4", "1p1q", "1p1q", "1p22", "1pzo", "1q1g", "1q4l",
    "1q8u", "1qf0", "1r1h", "1r1w", "1r55", "1r58", "1rdq", "1rth",
    "1s38", "1s3v", "1sg0", "1sqn", "1svh", "1syi", "1t46", "1t4e",
    "1t9e", "1tg6", "1tiy", "1tow", "1tt1", "1u1c", "1u4d", "1ual",
    "1ue8", "1uml", "1uou", "1uwt", "1uz1", "1v0p", "1v4s", "1vso",
    "1w1p", "1w6y", "1w7x", "1wbs", "1wc1", "1wnc", "1x38", "1x8d",
    "1xd0", "1xdd", "1xmb", "1xoz", "1y0l", "1y6a", "1ydr", "1ydt",
    "1ygc", "1yqj", "1ywn", "1z6e", "1zkk", "1zls", "2a0c", "2agl",
]


def download_pdb(pdb_id, output_dir):
    """Download PDB file from RCSB."""
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_file = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    if os.path.exists(output_file):
        return True
    
    try:
        urllib.request.urlretrieve(url, output_file)
        return True
    except Exception as e:
        print(f"  Failed to download {pdb_id}: {e}")
        return False


def download_dude_receptor(target_name, output_dir):
    """Download DUD-E receptor PDB."""
    url = f"https://dude.docking.org/targets/{target_name}/receptor.pdb"
    output_file = os.path.join(output_dir, f"{target_name}.pdb")
    
    if os.path.exists(output_file):
        return True
    
    try:
        urllib.request.urlretrieve(url, output_file)
        time.sleep(0.5)  # Be respectful to server
        return True
    except Exception as e:
        print(f"  Failed to download {target_name}: {e}")
        return False


def main():
    print("=" * 60)
    print("BENCHMARK DATASET EXPANSION")
    print("=" * 60)
    
    base_dir = Path("external_data/benchmark_raw")
    
    # Create directories
    dude_dir = base_dir / "dude"
    moad_dir = base_dir / "moad"
    crypto_dir = base_dir / "cryptobench"
    
    for d in [dude_dir, moad_dir, crypto_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Download DUD-E targets
    print("\n[1] Downloading DUD-E targets...")
    dude_success = 0
    for target in DUDE_TARGETS[:50]:  # Limit to 50 for initial test
        if download_dude_receptor(target, dude_dir):
            dude_success += 1
            print(f"  Downloaded {target} ({dude_success}/50)")
    
    print(f"    DUD-E: {dude_success} targets downloaded")
    
    # Download MOAD PDBs
    print("\n[2] Downloading Binding MOAD structures...")
    moad_success = 0
    for pdb_id in MOAD_QUALITY_PDBS[:50]:  # Limit to 50
        if download_pdb(pdb_id, moad_dir):
            moad_success += 1
            print(f"  Downloaded {pdb_id} ({moad_success}/50)")
    
    print(f"    MOAD: {moad_success} structures downloaded")
    
    # Download CryptoBench PDBs
    print("\n[3] Downloading CryptoBench structures...")
    crypto_success = 0
    for pdb_id in CRYPTOBENCH_PDBS[:50]:  # Limit to 50
        if download_pdb(pdb_id, crypto_dir):
            crypto_success += 1
            print(f"  Downloaded {pdb_id} ({crypto_success}/50)")
    
    print(f"    CryptoBench: {crypto_success} structures downloaded")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"DUD-E targets:     {dude_success:>5} downloaded to {dude_dir}")
    print(f"MOAD structures:   {moad_success:>5} downloaded to {moad_dir}")
    print(f"CryptoBench:       {crypto_success:>5} downloaded to {crypto_dir}")
    print("\nNext step: Run preprocessing to convert to PyG format")
    print("=" * 60)


if __name__ == "__main__":
    main()
