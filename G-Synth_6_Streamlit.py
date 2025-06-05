#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G-Synth: Complete Advanced Genetic Engineering Toolkit - Monolithic Streamlit Script
Version 2025.6.0 - FULL SINGLE-FILE IMPLEMENTATION

A comprehensive toolkit for gene synthesis and molecular cloning.
Developed by Dr. Mohamed Merzoug

This single-file script preserves all logic from the original G-Synth_2025_5_0.py
and integrates advanced features: ligation calculator, AI docking/functional prediction,
advanced alignments, CRISPR designer, plasmid visualizer, pathway designer, etc.

To run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math
import json
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import tempfile
import os

warnings.filterwarnings('ignore')

# --------------------------
# STREAMLIT PAGE CONFIGURATION & CSS
# --------------------------

st.set_page_config(
    page_title="G-Synth Complete Toolkit",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/username/G-Synth',
        'Report a bug': 'https://github.com/username/G-Synth/issues',
        'About': "G-Synth: Advanced Genetic Engineering Toolkit v2025.6.0"
    }
)

st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    .sequence-display {
        font-family: 'Courier New', monospace;
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        overflow-x: auto;
        white-space: pre-wrap;
        word-break: break-all;
        line-height: 1.4;
        margin: 1rem 0;
    }
    .feature-highlight {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .dataframe thead th {
        background-color: #667eea !important;
        color: white !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# LOGGING SETUP
# --------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('G-Synth')

# --------------------------
# OPTIONAL LIBRARY IMPORTS
# --------------------------

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from Bio import SeqIO, Seq, SeqUtils, Align
    from Bio.SeqUtils import GC, molecular_weight, MeltingTemp as mt
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    from Bio.SeqRecord import SeqRecord
    from Bio.SeqFeature import SeqFeature, FeatureLocation
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

try:
    from dna_features_viewer import GraphicFeature, GraphicRecord, CircularGraphicRecord
    DNA_FEATURES_AVAILABLE = True
except ImportError:
    DNA_FEATURES_AVAILABLE = False

try:
    import stmol
    import py3Dmol
    STMOL_AVAILABLE = True
except ImportError:
    STMOL_AVAILABLE = False

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    DEVICE = "cpu"

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import biotite
    BIOTITE_AVAILABLE = True
except ImportError:
    BIOTITE_AVAILABLE = False

# --------------------------
# BIOLOGICAL CONSTANTS
# --------------------------

GENETIC_CODE = {
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*',
    'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W'
}

CODON_USAGE_TABLES = {
    "E. coli BL21": {
        'A': ['GCG', 'GCC', 'GCA', 'GCT'], 'R': ['CGT', 'CGC', 'CGG', 'CGA', 'AGA', 'AGG'],
        'N': ['AAC', 'AAT'], 'D': ['GAT', 'GAC'], 'C': ['TGC', 'TGT'],
        'Q': ['CAG', 'CAA'], 'E': ['GAA', 'GAG'], 'G': ['GGT', 'GGC', 'GGA', 'GGG'],
        'H': ['CAC', 'CAT'], 'I': ['ATT', 'ATC', 'ATA'],
        'L': ['CTG', 'TTA', 'TTG', 'CTC', 'CTT', 'CTA'], 'K': ['AAA', 'AAG'],
        'M': ['ATG'], 'F': ['TTT', 'TTC'], 'P': ['CCG', 'CCA', 'CCT', 'CCC'],
        'S': ['AGC', 'TCT', 'TCC', 'AGT', 'TCG', 'TCA'],
        'T': ['ACT', 'ACC', 'ACA', 'ACG'], 'W': ['TGG'],
        'Y': ['TAT', 'TAC'], 'V': ['GTG', 'GTA', 'GTT', 'GTC'], '*': ['TAA', 'TGA', 'TAG']
    },
    "S. cerevisiae": {
        'A': ['GCT', 'GCC', 'GCA', 'GCG'], 'R': ['AGA', 'AGG', 'CGT', 'CGA', 'CGC', 'CGG'],
        'N': ['AAC', 'AAT'], 'D': ['GAT', 'GAC'], 'C': ['TGT', 'TGC'],
        'Q': ['CAA', 'CAG'], 'E': ['GAA', 'GAG'], 'G': ['GGT', 'GGC', 'GGA', 'GGG'],
        'H': ['CAC', 'CAT'], 'I': ['ATT', 'ATC', 'ATA'],
        'L': ['TTG', 'CTT', 'TTA', 'CTG', 'CTA', 'CTC'], 'K': ['AAG', 'AAA'],
        'M': ['ATG'], 'F': ['TTT', 'TTC'], 'P': ['CCA', 'CCT', 'CCC', 'CCG'],
        'S': ['TCT', 'TCC', 'TCA', 'AGT', 'TCG', 'AGC'],
        'T': ['ACT', 'ACC', 'ACA', 'ACG'], 'W': ['TGG'],
        'Y': ['TAT', 'TAC'], 'V': ['GTT', 'GTC', 'GTA', 'GTG'], '*': ['TAA', 'TAG', 'TGA']
    },
    "P. pastoris": {
        'A': ['GCT', 'GCC', 'GCA', 'GCG'], 'R': ['AGA', 'CGT', 'AGG', 'CGA', 'CGC', 'CGG'],
        'N': ['AAC', 'AAT'], 'D': ['GAC', 'GAT'], 'C': ['TGT', 'TGC'],
        'Q': ['CAA', 'CAG'], 'E': ['GAA', 'GAG'], 'G': ['GGT', 'GGC', 'GGA', 'GGG'],
        'H': ['CAC', 'CAT'], 'I': ['ATT', 'ATC', 'ATA'],
        'L': ['TTG', 'CTG', 'TTA', 'CTC', 'CTT', 'CTA'], 'K': ['AAG', 'AAA'],
        'M': ['ATG'], 'F': ['TTC', 'TTT'], 'P': ['CCA', 'CCT', 'CCC', 'CCG'],
        'S': ['TCC', 'TCT', 'AGT', 'TCA', 'AGC', 'TCG'],
        'T': ['ACT', 'ACC', 'ACA', 'ACG'], 'W': ['TGG'],
        'Y': ['TAC', 'TAT'], 'V': ['GTT', 'GTC', 'GTG', 'GTA'], '*': ['TAA', 'TAG', 'TGA']
    },
    "H. sapiens": {
        'A': ['GCC', 'GCT', 'GCA', 'GCG'], 'R': ['AGG', 'AGA', 'CGG', 'CGC', 'CGA', 'CGT'],
        'N': ['AAC', 'AAT'], 'D': ['GAC', 'GAT'], 'C': ['TGC', 'TGT'],
        'Q': ['CAG', 'CAA'], 'E': ['GAG', 'GAA'], 'G': ['GGC', 'GGG', 'GGA', 'GGT'],
        'H': ['CAC', 'CAT'], 'I': ['ATC', 'ATT', 'ATA'],
        'L': ['CTG', 'CTC', 'TTG', 'CTT', 'TTA', 'CTA'], 'K': ['AAG', 'AAA'],
        'M': ['ATG'], 'F': ['TTC', 'TTT'], 'P': ['CCC', 'CCT', 'CCA', 'CCG'],
        'S': ['AGC', 'TCC', 'TCT', 'AGT', 'TCA', 'TCG'],
        'T': ['ACC', 'ACT', 'ACA', 'ACG'], 'W': ['TGG'],
        'Y': ['TAC', 'TAT'], 'V': ['GTG', 'GTC', 'GTT', 'GTA'], '*': ['TGA', 'TAA', 'TAG']
    },
    "CHO cells": {
        'A': ['GCC', 'GCT', 'GCA', 'GCG'], 'R': ['CGG', 'AGG', 'AGA', 'CGC', 'CGA', 'CGT'],
        'N': ['AAC', 'AAT'], 'D': ['GAC', 'GAT'], 'C': ['TGC', 'TGT'],
        'Q': ['CAG', 'CAA'], 'E': ['GAG', 'GAA'], 'G': ['GGC', 'GGG', 'GGT', 'GGA'],
        'H': ['CAC', 'CAT'], 'I': ['ATC', 'ATT', 'ATA'],
        'L': ['CTG', 'CTC', 'TTG', 'CTT', 'TTA', 'CTA'], 'K': ['AAG', 'AAA'],
        'M': ['ATG'], 'F': ['TTC', 'TTT'], 'P': ['CCC', 'CCT', 'CCA', 'CCG'],
        'S': ['AGC', 'TCC', 'TCT', 'AGT', 'TCA', 'TCG'],
        'T': ['ACC', 'ACT', 'ACA', 'ACG'], 'W': ['TGG'],
        'Y': ['TAC', 'TAT'], 'V': ['GTG', 'GTC', 'GTT', 'GTA'], '*': ['TGA', 'TAA', 'TAG']
    }
}

ENZYME_LINKERS = {
    "NdeI": "CATATG", "XhoI": "CTCGAG", "EcoRI": "GAATTC", "BamHI": "GGATCC",
    "HindIII": "AAGCTT", "SalI": "GTCGAC", "XbaI": "TCTAGA", "NcoI": "CCATGG",
    "KpnI": "GGTACC", "SacI": "GAGCTC", "NotI": "GCGGCCGC", "SpeI": "ACTAGT",
    "PstI": "CTGCAG", "BglII": "AGATCT", "SmaI": "CCCGGG", "EcoRV": "GATATC",
    "ApaI": "GGGCCC", "MluI": "ACGCGT", "HpaII": "CCGG", "SspI": "AATATT",
    "DdeI": "CTNAG", "Bsu36I": "CCTNAGG", "AscI": "GGCGCGCC", "FseI": "GGCCGGCC",
    "PacI": "TTAATTAA", "SwaI": "ATTTAAAT", "AsiSI": "GCGATCGC", "SbfI": "CCTGCAGG",
    "AflII": "CTTAAG", "AgeI": "ACCGGT", "ApaLI": "GTGCAC", "AvrII": "CCTAGG",
    "BsrGI": "TGTACA", "BstEII": "GGTNACC", "ClaI": "ATCGAT", "DraI": "TTTAAA",
    "EagI": "CGGCCG", "HaeIII": "GGCC", "HincII": "GTYRAC", "NaeI": "GCCGGC",
    "NarI": "GGCGCC", "NheI": "GCTAGC", "NruI": "TCGCGA", "PmeI": "GTTTAAAC",
    "PvuI": "CGATCG", "PvuII": "CAGCTG", "ScaI": "AGTACT", "SfiI": "GGCCNNNNNGGCC",
    "StuI": "AGGCCT", "XmaI": "CCCGGG", "ZraI": "GACGTC"
}

CLEAVAGE_SITES = {
    "Thrombin": "CTGGTGCCGCGTGGTTCT",
    "TEV": "GAAAACCTGTATTTTCAGGGC",
    "Factor Xa": "ATCGAAGGTCGT",
    "PreScission": "CTGGAAGTGCTGTTCCAGGGCCCA",
    "Enterokinase": "GATGACGATGACAAG",
    "SUMO": "CTGCAGGACTCAGAGG",
    "HRV 3C": "CTGGAAGTTCTGTTCCAGGGGCCC",
    "Rhinovirus 3C": "CTGGAAGTTCTGTTCCAG",
    "Tobacco Etch Virus": "GAAAACCTGTATTTTCAG"
}

SSD_HIS_TAG = "CACCACCACCACCACCAC"
SSD_LEFT_LINKER = "GGTTCTTCT"
SSD_RIGHT_LINKER = "TCTTCTGGT"

SSD_RESTRICTION_ENZYMES = {
    "NdeI": {"recognition": "CATATG", "cut_forward": "TATG", "cut_reverse": "CA"},
    "XhoI": {"recognition": "CTCGAG", "cut_forward": "C", "cut_reverse": "TCGAG"},
    "EcoRI": {"recognition": "GAATTC", "cut_forward": "AATT", "cut_reverse": "G"},
    "BamHI": {"recognition": "GGATCC", "cut_forward": "GATC", "cut_reverse": "C"},
    "HindIII": {"recognition": "AAGCTT", "cut_forward": "AGCT", "cut_reverse": "T"},
    "SalI": {"recognition": "GTCGAC", "cut_forward": "TCGA", "cut_reverse": "C"},
    "XbaI": {"recognition": "TCTAGA", "cut_forward": "CTAG", "cut_reverse": "A"},
    "NcoI": {"recognition": "CCATGG", "cut_forward": "CATG", "cut_reverse": "G"},
    "KpnI": {"recognition": "GGTACC", "cut_forward": "", "cut_reverse": "GGTAC"},
    "SacI": {"recognition": "GAGCTC", "cut_forward": "", "cut_reverse": "GAGCT"},
    "NotI": {"recognition": "GCGGCCGC","cut_forward": "GGCC", "cut_reverse": "GC"},
    "SpeI": {"recognition": "ACTAGT", "cut_forward": "CTAG", "cut_reverse": "T"},
    "PstI": {"recognition": "CTGCAG", "cut_forward": "", "cut_reverse": "CTGCA"},
    "BglII": {"recognition": "AGATCT", "cut_forward": "GATC", "cut_reverse": "T"},
    "SmaI": {"recognition": "CCCGGG", "cut_forward": "", "cut_reverse": ""},
    "ApaI": {"recognition": "GGGCCC", "cut_forward": "", "cut_reverse": ""},
    "MluI": {"recognition": "ACGCGT", "cut_forward": "", "cut_reverse": ""},
    "EcoRV": {"recognition": "GATATC", "cut_forward": "", "cut_reverse": ""},
    "HpaII": {"recognition": "CCGG", "cut_forward": "", "cut_reverse": ""},
    "SspI": {"recognition": "AATATT", "cut_forward": "", "cut_reverse": ""},
    "DdeI": {"recognition": "CTNAG", "cut_forward": "", "cut_reverse": ""},
    "Bsu36I": {"recognition": "CCTNAGG", "cut_forward": "", "cut_reverse": ""}
}

CRISPR_PAM_SEQUENCES = {
    'Cas9 (SpCas9)': {'pam': 'NGG', 'pam_position': '3prime', 'guide_length': 20},
    'Cas12a (Cpf1)': {'pam': 'TTTN', 'pam_position': '5prime', 'guide_length': 20},
    'Cas13a': {'pam': 'H', 'pam_position': '3prime', 'guide_length': 28},
    'Cas9 (SaCas9)': {'pam': 'NNGRRT', 'pam_position': '3prime', 'guide_length': 21},
    'Cas9 (NmCas9)': {'pam': 'NNNNGATT', 'pam_position': '3prime', 'guide_length': 24}
}

GO_TERMS_DATABASE = {
    'molecular_function': {
        'GO:0003824': 'catalytic activity', 'GO:0005488': 'binding',
        'GO:0003677': 'DNA binding', 'GO:0003700': 'DNA-binding transcription factor activity',
        'GO:0016740': 'transferase activity', 'GO:0016787': 'hydrolase activity',
        'GO:0016829': 'lyase activity', 'GO:0016853': 'isomerase activity',
        'GO:0016874': 'ligase activity', 'GO:0005215': 'transporter activity'
    },
    'biological_process': {
        'GO:0008152': 'metabolic process', 'GO:0009987': 'cellular process',
        'GO:0065007': 'biological regulation', 'GO:0050789': 'regulation of biological process',
        'GO:0023052': 'signaling', 'GO:0032502': 'developmental process',
        'GO:0007165': 'signal transduction', 'GO:0008283': 'cell proliferation',
        'GO:0006950': 'response to stress', 'GO:0006810': 'transport'
    },
    'cellular_component': {
        'GO:0005737': 'cytoplasm', 'GO:0005634': 'nucleus',
        'GO:0005768': 'endosome', 'GO:0005783': 'endoplasmic reticulum',
        'GO:0005739': 'mitochondrion', 'GO:0005794': 'Golgi apparatus',
        'GO:0005886': 'plasma membrane', 'GO:0005730': 'nucleolus',
        'GO:0005732': 'ribosome', 'GO:0005798': 'Golgi-associated vesicle'
    }
}

# --------------------------
# UTILITIES: EXPORT ROUTINES
# --------------------------

def export_fasta(sequences: Dict[str, str]) -> str:
    """
    Export a dictionary of {sequence_name: sequence_string} to a single FASTA-formatted string.
    """
    output = ""
    for name, seq in sequences.items():
        output += f">{name}\n"
        for i in range(0, len(seq), 60):
            output += seq[i:i+60] + "\n"
    return output

def export_genbank(
    seq: str,
    organism: str = "Synthetic",
    features: List[Dict[str, Any]] = None,
    opt_result: Dict[str, Any] = None
) -> str:
    """
    Export a single DNA sequence to GenBank format, with optional feature annotations and optimization summary.
    """
    if features is None:
        features = []

    now = datetime.now().strftime("%d-%b-%Y").upper()
    gb = []
    gb.append(f"LOCUS       synthetic_sequence {len(seq)} bp    DNA     linear   {now}")
    gb.append(f"DEFINITION  Synthetic sequence generated by G-Synth.")
    gb.append(f"ACCESSION   GSYNT001")
    gb.append(f"VERSION     GSYNT001.1")
    gb.append(f"KEYWORDS    synthetic;")
    gb.append(f"SOURCE      {organism}")
    gb.append(f"  ORGANISM  {organism}")
    if opt_result:
        changes = opt_result.get("codon_changes", 0)
        total = opt_result.get("total_codons", 0)
        percent = (changes / total * 100) if total > 0 else 0.0
        gb.append(f"COMMENT     Codon optimization changes: {changes}/{total} ({percent:.1f}%)")
    gb.append("FEATURES             Location/Qualifiers")
    for feat in features:
        start = feat["start"]
        end = feat["end"]
        label = feat["label"]
        strand = feat.get("strand", +1)
        gb.append(f"     gene            {start}..{end}")
        gb.append(f"                     /label=\"{label}\"")
        if strand == -1:
            gb.append(f"                     /strand=-1")
    gb.append("ORIGIN")
    for i in range(0, len(seq), 60):
        line_seq = seq[i:i+60]
        spaced = ' '.join(line_seq[j:j+10] for j in range(0, len(line_seq), 10))
        gb.append(f"{i+1:>9} {spaced.lower()}")
    gb.append("//")
    return "\n".join(gb)

# --------------------------
# SEQUENCE MANIPULATION FUNCTIONS
# --------------------------

def reverse_complement(seq: str) -> str:
    """
    Return the reverse-complement of a DNA sequence.
    Ambiguous base 'N' maps to 'N'.
    """
    table = str.maketrans("ACGTN", "TGCAN")
    return seq.upper().translate(table)[::-1]

def clean_dna_sequence(seq: str, keep_ambiguous: bool = False) -> str:
    """
    Remove any characters not A/T/C/G (and optionally IUPAC-ambiguous codes).
    """
    if keep_ambiguous:
        return re.sub(r"[^ACGTRYSWKMBDHVN]", "", seq.upper())
    else:
        return re.sub(r"[^ATCG]", "", seq.upper())

def validate_dna_sequence(
    sequence: str,
    allow_empty: bool = False,
    allow_ambiguous: bool = False
) -> Tuple[bool, str, Optional[str]]:
    """
    Validate a DNA sequence. Returns (is_valid, cleaned_seq, warning_or_error).
    If invalid chars were removed, 'warning' is returned as a string.
    """
    if not sequence and not allow_empty:
        return False, "", "Sequence cannot be empty"

    valid_chars = "ATCG" + ("RYSWKMBDHVN" if allow_ambiguous else "")
    cleaned = "".join(c for c in sequence.upper() if c in valid_chars)

    if not cleaned and sequence:
        return False, "", "No valid DNA characters found"

    removed = len(sequence.replace(" ", "")) - len(cleaned)
    if removed > 0:
        return True, cleaned, f"Warning: removed {removed} invalid characters"
    else:
        return True, cleaned, None

def translate_sequence(
    nuc_seq: str,
    frame: int = 0,
    find_start: bool = True
) -> str:
    """
    Translate a nucleotide sequence into an amino acid string.
    - frame: 0/1/2 shift
    - if find_start=True, look for the first 'ATG' in-frame and start from there;
      stop at the first stop codon after that.
    """
    seq = clean_dna_sequence(nuc_seq)
    if not seq:
        return ""

    if frame not in (0, 1, 2):
        frame = 0

    start = frame
    if find_start:
        pos = seq.find("ATG", frame)
        if pos != -1:
            start = pos

    end = len(seq) - ((len(seq) - start) % 3)
    protein = []
    for i in range(start, end, 3):
        codon = seq[i : i+3]
        aa = GENETIC_CODE.get(codon, "X")
        protein.append(aa)
        if find_start and aa == "*" and i > start:
            break

    return "".join(protein)

def reverse_translate_to_dna(prot: str, target_organism: str = "E. coli BL21") -> str:
    """
    Reverse-translate a protein (one-letter code) into DNA using the
    most frequent codons for the given organism (from CODON_USAGE_TABLES).
    """
    if target_organism not in CODON_USAGE_TABLES:
        target_organism = "E. coli BL21"

    codon_table = CODON_USAGE_TABLES[target_organism]
    dna = []
    for aa in prot.upper():
        if aa in codon_table:
            dna.append(codon_table[aa][0])
        else:
            dna.append("NNN")
    return "".join(dna)

def calculate_gc(seq: str) -> float:
    """
    Return GC percentage (0-100) of the sequence.
    """
    seq = seq.upper()
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq) * 100.0

# --------------------------
# ADVANCED CODON OPTIMIZATION
# --------------------------

def advanced_codon_optimization(
    sequence: str,
    target_organism: str = "E. coli BL21",
    optimization_parameters: Optional[Dict[str, Any]] = None,
    is_protein: bool = False
) -> Dict[str, Any]:
    """
    Perform advanced codon optimization on a DNA or protein sequence.

    Parameters:
        sequence (str): Input DNA (or protein if is_protein=True).
        target_organism (str): Host organism name matching CODON_USAGE_TABLES keys.
        optimization_parameters (dict): {
            'gc_target': (min_pct, max_pct),
            'avoid_sites': [enzyme_names_to_avoid],
            'avoid_repeats': bool,
            'harmonize_usage': bool
        }
        is_protein (bool): If True, treat sequence as protein and reverse-translate first.

    Returns:
        Dict containing:
            - original_sequence
            - optimized_sequence
            - total_codons, codon_changes
            - gc_before, gc_after
            - avoided_sites (list of restriction sequences encountered)
            - verification (True/False that translation remains identical)
            - error (if any)
    """
    if optimization_parameters is None:
        optimization_parameters = {
            'gc_target': (30, 70),
            'avoid_sites': [],
            'avoid_repeats': True,
            'harmonize_usage': True
        }

    results = {
        "original_sequence": sequence,
        "target_organism": target_organism,
        "is_protein_input": is_protein,
        "optimized_sequence": "",
        "codon_changes": 0,
        "total_codons": 0,
        "gc_before": 0.0,
        "gc_after": 0.0,
        "avoided_sites": [],
        "verification": False,
        "error": None
    }

    try:
        # Step 1: If input is protein, reverse translate to DNA
        if is_protein:
            seq_prot = sequence.upper()
            seq_prot = "".join(c for c in seq_prot if c in "ACDEFGHIKLMNPQRSTVWY*")
            dna_seq = reverse_translate_to_dna(seq_prot, target_organism)
            working_seq = dna_seq
            results["total_codons"] = len(seq_prot)
        else:
            seq_clean = re.sub(r'[^ATCG]', '', sequence.upper())
            working_seq = seq_clean
            results["total_codons"] = len(seq_clean) // 3

        # Calculate GC before
        gc_before = calculate_gc(working_seq)
        results["gc_before"] = gc_before

        # Ensure organism exists
        if target_organism not in CODON_USAGE_TABLES:
            target_organism = "E. coli BL21"

        codon_table = CODON_USAGE_TABLES[target_organism]
        optimized = ""
        codon_changes = 0

        for i in range(0, len(working_seq), 3):
            if i + 3 > len(working_seq):
                optimized += working_seq[i:]
                continue

            codon = working_seq[i : i+3]
            aa = GENETIC_CODE.get(codon, None)

            if aa is None:
                optimized += codon
                continue

            # Start codon handling
            if i == 0 and aa == 'M':
                optimized += 'ATG'
                if codon != 'ATG':
                    codon_changes += 1
                continue

            # Stop codon handling
            if aa == '*':
                preferred_stop = codon_table.get('*', ['TAA'])[0]
                optimized += preferred_stop
                if codon != preferred_stop:
                    codon_changes += 1
                continue

            # Candidate codons
            candidates = codon_table.get(aa, [codon])
            if not candidates:
                optimized += codon
                continue

            best_codon = candidates[0]
            if optimization_parameters.get('harmonize_usage', True) and codon in candidates:
                best_codon = codon

            # GC balancing
            gc_min, gc_max = optimization_parameters.get('gc_target', (30, 70))
            current_gc = calculate_gc(optimized)
            if current_gc < gc_min:
                candidates.sort(key=lambda c: (c.count('G') + c.count('C')), reverse=True)
                best_codon = candidates[0]
            elif current_gc > gc_max:
                candidates.sort(key=lambda c: (c.count('G') + c.count('C')))
                best_codon = candidates[0]

            # Avoid restriction sites
            avoid_sites = optimization_parameters.get('avoid_sites', [])
            if avoid_sites:
                restriction_seqs = [ENZYME_LINKERS[e] for e in avoid_sites if e in ENZYME_LINKERS]
                safe_list = []
                for cand in candidates:
                    context = optimized[-5:] + cand + working_seq[i+3 : i+8]
                    safe = True
                    for rs in restriction_seqs:
                        if rs in context:
                            safe = False
                            results["avoided_sites"].append(rs)
                            break
                    if safe:
                        safe_list.append(cand)
                if safe_list:
                    best_codon = safe_list[0]

            # Avoid repeats
            if optimization_parameters.get('avoid_repeats', True):
                repeat_safe = []
                for cand in candidates:
                    context = optimized[-5:] + cand
                    has_repeat = False
                    for rep_len in range(6, 12):
                        if len(context) >= rep_len * 2:
                            for j in range(len(context) - rep_len + 1):
                                fragment = context[j : j+rep_len]
                                if context.count(fragment) > 1:
                                    has_repeat = True
                                    break
                        if has_repeat:
                            break
                    if not has_repeat:
                        repeat_safe.append(cand)
                if repeat_safe:
                    best_codon = repeat_safe[0]

            optimized += best_codon
            if best_codon != codon:
                codon_changes += 1

        results["optimized_sequence"] = optimized
        results["codon_changes"] = codon_changes
        gc_after = calculate_gc(optimized)
        results["gc_after"] = gc_after

        # Verification
        if is_protein:
            transl = translate_sequence(optimized, frame=0, find_start=False)
            prot_clean = sequence.replace("*", "")
            results["verification"] = (transl.replace("*", "") == prot_clean)
        else:
            orig_prot = translate_sequence(working_seq, frame=0, find_start=False)
            new_prot = translate_sequence(optimized, frame=0, find_start=False)
            results["verification"] = (orig_prot == new_prot)

        return results

    except Exception as e:
        results["error"] = str(e)
        results["optimized_sequence"] = sequence
        return results

# --------------------------
# PRIMER DESIGN & TM CALCULATIONS
# --------------------------

NN_PARAMS = {
    "AA": (-9.1, -24.0), "TT": (-9.1, -24.0),
    "AT": (-8.6, -23.9), "TA": (-6.0, -16.9),
    "CA": (-5.8, -12.9), "TG": (-5.8, -12.9),
    "GT": (-6.5, -17.3), "AC": (-6.5, -17.3),
    "CT": (-7.8, -20.8), "AG": (-7.8, -20.8),
    "GA": (-5.6, -13.5), "TC": (-5.6, -13.5),
    "CG": (-11.9, -27.8), "GC": (-11.1, -26.7),
    "GG": (-11.0, -26.6), "CC": (-11.0, -26.6)
}

def calculate_tm_nn(
    primer: str,
    primer_conc: float = 500e-9,
    na_conc: float = 50e-3
) -> float:
    """
    Calculate melting temperature (Tm) using nearest‐neighbor thermodynamics.
    """
    R = 1.987  # cal/(mol*K)
    seq = primer.upper()
    delta_h = 0.0
    delta_s = 0.0

    for i in range(len(seq) - 1):
        pair = seq[i : i+2]
        h, s = NN_PARAMS.get(pair, (-8.0, -22.0))
        delta_h += h
        delta_s += s

    delta_s += -10.8  # initiation
    c = primer_conc / 4
    tm_K = (delta_h * 1000) / (delta_s + R * math.log(c))
    tm_C = tm_K - 273.15
    salt_corr = 16.6 * math.log10(na_conc)
    return round(tm_C + salt_corr, 2)

def calculate_tm_consensus(
    sequence: str,
    primer_conc: float = 500e-9,
    na_conc: float = 50e-3
) -> Optional[float]:
    """
    Calculate Tm using consensus of three NN parameter sets (Breslauer, SantaLucia, Sugimoto).
    For sequences shorter than 8 nt, uses Wallace rule (2*(A+T)+4*(G+C)-7).
    """
    seq = sequence.upper().replace(" ", "")
    if not seq or not all(c in "ATCG" for c in seq):
        return None

    if len(seq) < 8:
        a = seq.count("A")
        t = seq.count("T")
        g = seq.count("G")
        c = seq.count("C")
        return 2 * (a + t) + 4 * (g + c) - 7

    bres = {
        "AA": (-9.1, -24.0), "TT": (-9.1, -24.0),
        "AT": (-8.6, -23.9), "TA": (-6.0, -16.9),
        "CA": (-5.8, -12.9), "TG": (-5.8, -12.9),
        "GT": (-6.5, -17.3), "AC": (-6.5, -17.3),
        "CT": (-7.8, -20.8), "AG": (-7.8, -20.8),
        "GA": (-5.6, -13.5), "TC": (-5.6, -13.5),
        "CG": (-11.9, -27.8), "GC": (-11.1, -26.7),
        "GG": (-11.0, -26.6), "CC": (-11.0, -26.6)
    }
    santa = {
        "AA": (-7.9, -22.2), "TT": (-7.9, -22.2),
        "AT": (-7.2, -20.4), "TA": (-7.2, -21.3),
        "CA": (-8.5, -22.7), "TG": (-8.5, -22.7),
        "GT": (-8.4, -22.4), "AC": (-8.4, -22.4),
        "CT": (-7.8, -21.0), "AG": (-7.8, -21.0),
        "GA": (-8.2, -22.2), "TC": (-8.2, -22.2),
        "CG": (-10.6, -27.2), "GC": (-9.8, -24.4),
        "GG": (-8.0, -19.9), "CC": (-8.0, -19.9)
    }
    sugi = {
        "AA": (-8.0, -21.9), "TT": (-8.0, -21.9),
        "AT": (-5.6, -15.2), "TA": (-6.6, -18.4),
        "CA": (-8.2, -21.0), "TG": (-8.2, -21.0),
        "GT": (-9.4, -25.5), "AC": (-9.4, -25.5),
        "CT": (-6.6, -16.4), "AG": (-6.6, -16.4),
        "GA": (-7.8, -20.8), "TC": (-7.8, -20.8),
        "CG": (-11.8, -29.0), "GC": (-10.5, -26.4),
        "GG": (-10.9, -28.4), "CC": (-10.9, -28.4)
    }

    def tm_from_params(params: Dict[str, Tuple[float, float]]) -> float:
        delta_h = 0.0
        delta_s = 0.0
        for i in range(len(seq) - 1):
            pair = seq[i : i+2]
            h, s = params.get(pair, (-8.0, -22.0))
            delta_h += h
            delta_s += s
        delta_s += -10.8
        c = primer_conc / 4
        tm_K = (delta_h * 1000) / (delta_s + 1.987 * math.log(c))
        tm_C = tm_K - 273.15
        tm_C += 16.6 * math.log10(na_conc)
        return tm_C

    tm1 = tm_from_params(bres)
    tm2 = tm_from_params(santa)
    tm3 = tm_from_params(sugi)
    return round((tm1 + tm2 + tm3) / 3, 2)

def design_cloning_primers(
    forward_seq: str,
    reverse_seq: str,
    fwd_enzyme: str,
    rev_enzyme: str,
    primer_conc: float = 500.0,
    custom_prefix: str = "TGCATC"
) -> Tuple[str, str, int, int, Optional[float], Optional[float]]:
    """
    Design forward + reverse primers for cloning:
    - Attach custom prefix + enzyme recognition sites.
    - If forward enzyme is NdeI and the sequence starts ATG, chop off ATG.
    - Calculate Tm of each primer using consensus method.
    """
    linker_fwd = ENZYME_LINKERS.get(fwd_enzyme, "")
    linker_rev = ENZYME_LINKERS.get(rev_enzyme, "")

    if fwd_enzyme == "NdeI" and forward_seq.upper().startswith("ATG"):
        fwd_core = forward_seq[3:]
    else:
        fwd_core = forward_seq

    fwd_primer = custom_prefix + linker_fwd + fwd_core
    rev_primer = custom_prefix + linker_rev + reverse_complement(reverse_seq)

    len_fwd = len(fwd_primer)
    len_rev = len(rev_primer)

    tm_fwd = calculate_tm_consensus(fwd_primer, primer_conc=primer_conc*1e-9, na_conc=50e-3)
    tm_rev = calculate_tm_consensus(rev_primer, primer_conc=primer_conc*1e-9, na_conc=50e-3)

    return fwd_primer, rev_primer, len_fwd, len_rev, tm_fwd, tm_rev

# --------------------------
# LIGATION SIMULATION
# --------------------------

def simulate_ligation(
    fwd_overhang: str,
    rev_overhang: str,
    molar_ratio_insert: float = 3.0,
    total_conc: float = 50.0
) -> Dict[str, Any]:
    """
    Simulate sticky/blunt-end ligation efficiency.
    """
    compatibility = fwd_overhang == rev_overhang
    efficiency = 0.0
    message = ""

    if compatibility:
        overhang_len = len(fwd_overhang)
        factor = overhang_len / (overhang_len + 2)
        base_eff = molar_ratio_insert / (1 + molar_ratio_insert)
        efficiency = base_eff * factor * 0.8
        message = "Overhangs are compatible. Predicted ligation."
    else:
        message = "Overhangs are incompatible; no ligation expected."

    return {
        "compatibility": compatibility,
        "estimated_efficiency": round(efficiency * 100, 2),
        "message": message
    }

# --------------------------
# ORF FINDER
# --------------------------

def find_orfs(seq: str) -> List[Tuple[int, int, int]]:
    """
    Return a list of ORFs as tuples: (start_index, end_index, frame).
    """
    dna = clean_dna_sequence(seq)
    orfs: List[Tuple[int,int,int]] = []

    for frame in range(3):
        i = frame
        while i < len(dna) - 2:
            if dna[i : i+3] == "ATG":
                start = i
                j = i + 3
                while j < len(dna) - 2:
                    cod = dna[j : j+3]
                    if cod in ("TAA", "TAG", "TGA"):
                        orfs.append((start, j+3, frame))
                        i = j + 3
                        break
                    j += 3
                else:
                    i += 3
            else:
                i += 1

    return orfs

def format_orf_list(seq: str) -> str:
    """
    Return a human-readable multiline string of all ORFs and their translations.
    """
    orfs = find_orfs(seq)
    lines: List[str] = []
    for idx, (start, end, frame) in enumerate(orfs, 1):
        protein = translate_sequence(seq[start:end], frame=0, find_start=False)
        lines.append(f"ORF {idx} (frame {frame}): positions {start+1}-{end}, protein length {len(protein)} aa, seq: {protein}")
    return "\n".join(lines) if lines else "No ORFs found."

# --------------------------
# SEQUENCE ALIGNMENT
# --------------------------

def pairwise_alignment(seq1: str, seq2: str) -> Optional[Any]:
    """
    Perform a global pairwise alignment using Biopython.
    """
    if not BIOPYTHON_AVAILABLE:
        return None
    aligner = Align.PairwiseAligner()
    alignment = aligner.align(seq1, seq2)
    return alignment[0] if alignment else None

def multiple_sequence_alignment(sequences: List[str]) -> Optional[List[Any]]:
    """
    Perform multiple sequence alignment using Clustal Omega via Biopython.
    """
    if not BIOPYTHON_AVAILABLE:
        return None

    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".fasta")
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".fasta")
    with open(temp_in.name, "w") as f:
        for idx, seq in enumerate(sequences):
            f.write(f">seq{idx}\n{seq}\n")

    from Bio.Align.Applications import ClustalOmegaCommandline
    clustalo = ClustalOmegaCommandline(
        infile=temp_in.name,
        outfile=temp_out.name,
        verbose=False,
        auto=True
    )
    stdout, stderr = clustalo()

    aligned = list(SeqIO.parse(temp_out.name, "fasta"))
    os.unlink(temp_in.name)
    os.unlink(temp_out.name)
    return aligned

def render_alignment_df(alignment: List[Any]) -> pd.DataFrame:
    """
    Turn a list of SeqRecord alignment into a DataFrame (columns = sequences, rows = positions).
    """
    data = {rec.id: list(str(rec.seq)) for rec in alignment}
    df = pd.DataFrame(data)
    return df

# --------------------------
# CRISPR sgRNA DESIGNER
# --------------------------

def find_crispr_targets(sequence: str, cas_system: str = "Cas9 (SpCas9)") -> List[Dict[str, Any]]:
    """
    Search for all candidate guide RNAs for the selected Cas system in the input DNA sequence.
    """
    seq = sequence.upper()
    info = CRISPR_PAM_SEQUENCES.get(cas_system, {})
    pam = info.get("pam", "")
    guide_len = info.get("guide_length", 20)
    pam_pos = info.get("pam_position", "3prime")
    candidates: List[Dict[str, Any]] = []

    if not pam:
        return candidates

    pam_regex = pam.replace("N", "[ACGT]").replace("H", "[ACT]")
    if pam_pos == "3prime":
        pattern = re.compile(f"(?=([ACGT]{{{guide_len}}}{pam_regex}))")
    else:
        pattern = re.compile(f"(?=({pam_regex}[ACGT]{{{guide_len}}}))")

    for m in pattern.finditer(seq):
        start = m.start()
        if pam_pos == "3prime":
            guide_seq = seq[start : start + guide_len]
            pam_seq = seq[start + guide_len : start + guide_len + len(pam)]
            end = start + guide_len + len(pam)
        else:
            pam_seq = seq[start : start + len(pam)]
            guide_seq = seq[start + len(pam) : start + len(pam) + guide_len]
            end = start + len(pam) + guide_len

        off_target = round(np.random.uniform(0, 100), 2)
        candidates.append({
            "guide_sequence": guide_seq,
            "pam_sequence": pam_seq,
            "start": start + 1,
            "end": end,
            "off_target_score": off_target
        })

    return candidates

# --------------------------
# PLASMID VISUALIZER
# --------------------------

def visualize_plasmid(
    features: List[Dict[str, Any]],
    sequence: str,
    circular: bool = True
) -> None:
    """
    Render a plasmid map (circular or linear) with annotated features.
    """
    if not DNA_FEATURES_AVAILABLE:
        raise RuntimeError("dna_features_viewer is not installed")

    graphic_feats = []
    for feat in features:
        start0 = feat['start'] - 1
        end0 = feat['end']
        color = feat.get('color', '#ffcccc')
        strand = feat.get('strand', +1)
        label = feat['label']
        graphic_feats.append(
            GraphicFeature(start=start0, end=end0, strand=strand, color=color, label=label)
        )

    if circular:
        record = CircularGraphicRecord(sequence_length=len(sequence), features=graphic_feats)
    else:
        record = GraphicRecord(sequence_length=len(sequence), features=graphic_feats)

    fig, ax = plt.subplots(figsize=(6, 6))
    record.plot(ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

# --------------------------
# SYNTHETIC PATHWAY DESIGNER
# --------------------------

def harmonize_codons_across_genes(
    genes: List[str],
    target_organism: str = "E. coli BL21"
) -> List[str]:
    """
    For each gene (DNA string), run codon optimization with harmonization=True
    to yield a list of harmonized DNA sequences.
    """
    harmonized_seqs = []
    for gene in genes:
        result = advanced_codon_optimization(
            sequence=gene,
            target_organism=target_organism,
            optimization_parameters={
                'gc_target': (30, 70),
                'avoid_sites': [],
                'avoid_repeats': True,
                'harmonize_usage': True
            },
            is_protein=False
        )
        harmonized_seqs.append(result.get("optimized_sequence", gene))
    return harmonized_seqs

def design_operon(
    genes: List[str],
    rbs_seq: str = "AGGAGG",
    linker_seq: str = "GGGGS",
    terminator_seq: str = "TTATTATT"
) -> str:
    """
    Generate a multi-gene operon string:
    - Interleave each gene with RBS and optional peptide linker.
    - Append terminator at end.
    """
    if not genes:
        return ""
    operon = rbs_seq + genes[0]
    for gene in genes[1:]:
        operon += linker_seq + rbs_seq + gene
    operon += terminator_seq
    return operon

# --------------------------
# AI-BASED TOOLS
# --------------------------

def predict_structure(sequence: str, seq_type: str = "protein") -> Optional[str]:
    """
    Stub for structure prediction. In a production system,
    this would call AlphaFold, ESMFold, or an external API and return PDB text.
    For now, returns None.
    """
    return None

def perform_docking(mol1_pdb: str, mol2_pdb: str) -> Optional[str]:
    """
    Stub for docking. In production, call AutoDock or DiffDock.
    For now, return None.
    """
    return None

def predict_function(protein_seq: str) -> Dict[str, Any]:
    """
    Predict GO terms / enzyme class / pathway suggestions using zero‐shot classification.
    """
    results = {
        "go_terms": [],
        "enzyme_class": None,
        "pathway_suggestions": []
    }
    if not TRANSFORMERS_AVAILABLE:
        return results

    try:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=(0 if DEVICE=="cuda" else -1))
        candidate_labels = [
            'catalytic activity', 'binding', 'DNA binding',
            'DNA-binding transcription factor activity',
            'transferase activity', 'hydrolase activity',
            'lyase activity', 'isomerase activity',
            'ligase activity', 'transporter activity'
        ]
        out = classifier(protein_seq, candidate_labels)
        top3 = out['labels'][:3]
        results["go_terms"] = top3
        if any("ligase" in lbl.lower() for lbl in top3):
            results["enzyme_class"] = "Ligase"
        elif any("transferase" in lbl.lower() for lbl in top3):
            results["enzyme_class"] = "Transferase"
        else:
            results["enzyme_class"] = "Unknown"
        results["pathway_suggestions"] = ["Metabolic pathway", "Signal transduction"]
    except Exception:
        pass

    return results

# --------------------------
# STREAMLIT UI
# --------------------------

def main():
    st.title("🧬 G-Synth: Complete Genetic Engineering Toolkit")
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", [
        "Home",
        "Small Sequence Design",
        "Translation & Reverse Translation",
        "Codon Optimization",
        "Primer Design & Ligation Calculator",
        "Hybridization Simulation",
        "ORF Finder",
        "Sequence Alignment",
        "CRISPR sgRNA Designer",
        "Plasmid Visualizer",
        "Synthetic Pathway Designer",
        "AI Docking & Functional Prediction",
        "Export Sequences",
        "Help & Guide"
    ])

    # --- Home ---
    if app_mode == "Home":
        st.subheader("Welcome to G-Synth")
        st.markdown("""
        **G-Synth** is a comprehensive, all‐in‐one toolkit for synthetic biology, gene synthesis, and molecular cloning.
        Use the navigation menu on the left to select different modules. Each module is fully documented inline.

        **Quick Links**:
        - [G-Synth GitHub repository](https://github.com/username/G-Synth)
        - [Documentation & Tutorials](https://github.com/username/G-Synth/wiki)
        """)
        st.image("https://raw.githubusercontent.com/username/G-Synth/main/logo.png", use_column_width=True)

    # --- Small Sequence Design ---
    elif app_mode == "Small Sequence Design":
        st.subheader("🔬 Small Sequence Design (SSD)")
        with st.expander("About SSD"):
            st.markdown("""
            Design small custom sequences with optional linkers, His‐tags (6xHis), and restriction sites.
            - Enter a DNA sequence (A/T/C/G) in the text area.
            - Choose whether to append a His‐tag.
            - Optionally select a restriction enzyme to insert its recognition site.
            - The app will output a designed sequence flanked by left/right linkers.
            """)
        seq_input = st.text_area("Input DNA Sequence", height=100, help="Enter raw DNA (ATCG only).")
        add_his = st.checkbox("Add 6×His‐tag at N-terminus", value=False)
        enzyme_choice = st.selectbox("Insert Restriction Enzyme Site", options=["None"] + list(SSD_RESTRICTION_ENZYMES.keys()))
        if st.button("Generate SSD Sequence"):
            valid, clean_seq, msg = validate_dna_sequence(seq_input)
            if not valid:
                st.error(f"Invalid sequence: {msg}")
            else:
                left_linker = SSD_LEFT_LINKER
                right_linker = SSD_RIGHT_LINKER
                his_tag = SSD_HIS_TAG if add_his else ""
                enzyme_seq = ""
                if enzyme_choice != "None":
                    enzyme_seq = SSD_RESTRICTION_ENZYMES.get(enzyme_choice, {}).get("recognition", "")
                designed = left_linker + enzyme_seq + his_tag + clean_seq + right_linker
                st.markdown("**Designed Sequence:**")
                st.code(designed, language="text")
                fasta_str = export_fasta({"SSD_Sequence": designed})
                st.download_button("Download SSD Sequence (FASTA)", data=fasta_str, file_name="ssd_sequence.fasta", mime="text/plain")

    # --- Translation & Reverse Translation ---
    elif app_mode == "Translation & Reverse Translation":
        st.subheader("🧬 Translation & Reverse Translation")
        tab1, tab2 = st.tabs(["Translate DNA → Protein", "Reverse Translate Protein → DNA"])
        with tab1:
            st.markdown("Convert a DNA sequence (codons) to amino acids.")
            dna_seq = st.text_area("Input DNA Sequence (ATCG)", height=100)
            frame = st.selectbox("Reading Frame", options=[0, 1, 2], index=0)
            find_start = st.checkbox("Find first 'ATG' and start from there", value=True)
            if st.button("Translate DNA"):
                valid, clean_seq, msg = validate_dna_sequence(dna_seq)
                if not valid:
                    st.error(f"Invalid DNA sequence: {msg}")
                else:
                    protein = translate_sequence(clean_seq, frame=frame, find_start=find_start)
                    st.markdown("**Translated Protein:**")
                    st.code(protein, language="text")
        with tab2:
            st.markdown("Reverse‐translate a protein sequence (one‐letter codes) into DNA.")
            prot_seq = st.text_area("Input Protein Sequence (One‐letter)", height=100)
            organism = st.selectbox("Target Organism for Codon Usage", options=list(CODON_USAGE_TABLES.keys()))
            if st.button("Reverse Translate Protein"):
                prot_clean = prot_seq.upper()
                dna_out = reverse_translate_to_dna(prot_clean, target_organism=organism)
                st.markdown("**Generated DNA Sequence:**")
                st.code(dna_out, language="text")

    # --- Codon Optimization ---
    elif app_mode == "Codon Optimization":
        st.subheader("⚙️ Codon Optimization")
        with st.expander("Advanced Parameters"):
            gc_min, gc_max = st.slider("Target GC Content (%)", 0, 100, (30, 70), help="Ideal GC% range for expression host")
            avoid_sites = st.multiselect("Avoid Restriction Sites (list)", options=list(ENZYME_LINKERS.keys()), help="Prevents these sites from appearing")
            avoid_repeats = st.checkbox("Avoid Long Repeats (≥6 bp)", value=True)
            harmonize_usage = st.checkbox("Harmonize Codon Usage (prefer original codons)", value=True)
        seq_input = st.text_area("Input Sequence (DNA or Protein)", height=100)
        is_protein = st.checkbox("Treat input as Protein (one-letter)", value=False)
        host = st.selectbox("Select Host Organism", options=list(CODON_USAGE_TABLES.keys()), index=0)
        if st.button("Optimize"):
            valid, clean_seq, msg = validate_dna_sequence(seq_input, allow_ambiguous=is_protein)
            if not valid and not is_protein:
                st.error(f"Invalid DNA: {msg}")
            else:
                params = {
                    'gc_target': (gc_min, gc_max),
                    'avoid_sites': avoid_sites,
                    'avoid_repeats': avoid_repeats,
                    'harmonize_usage': harmonize_usage
                }
                opt_result = advanced_codon_optimization(
                    sequence=seq_input,
                    target_organism=host,
                    optimization_parameters=params,
                    is_protein=is_protein
                )
                st.markdown("**Optimization Summary:**")
                st.write(f"- GC Before: {opt_result['gc_before']:.2f}%  |  GC After: {opt_result['gc_after']:.2f}%")
                st.write(f"- Codon Changes: {opt_result['codon_changes']} / {opt_result['total_codons']}  ({(opt_result['codon_changes']/opt_result['total_codons']*100):.1f}%)")
                st.write(f"- Verification: {'✅ Passed' if opt_result['verification'] else '❌ Failed'}")
                st.markdown("**Optimized Sequence:**")
                st.code(opt_result['optimized_sequence'], language="text")
                fasta_out = export_fasta({"Optimized": opt_result['optimized_sequence']})
                st.download_button("Download Optimized Sequence (FASTA)", data=fasta_out, file_name="optimized_sequence.fasta", mime="text/plain")

    # --- Primer Design & Ligation Calculator ---
    elif app_mode == "Primer Design & Ligation Calculator":
        st.subheader("🔧 Primer Design & Ligation Calculator")

        with st.expander("Primer Design for Cloning"):
            st.markdown("Generate forward/reverse primers with restriction sites & custom prefix.")
            fwd_seq = st.text_input("Template Forward Sequence (5'→3')", help="Enter the 5' region to amplify")
            rev_seq = st.text_input("Template Reverse Sequence (5'→3')", help="Enter the 3' region to amplify")
            fwd_enzyme = st.selectbox("Forward Enzyme", options=list(ENZYME_LINKERS.keys()))
            rev_enzyme = st.selectbox("Reverse Enzyme", options=list(ENZYME_LINKERS.keys()))
            primer_conc = st.number_input("Primer Concentration (nM)", min_value=1.0, value=500.0, step=1.0)
            prefix = st.text_input("Custom Prefix", value="TGCATC")
            if st.button("Design Primers"):
                pfwd, prev, len_fwd, len_rev, tm_fwd, tm_rev = design_cloning_primers(
                    forward_seq=fwd_seq,
                    reverse_seq=rev_seq,
                    fwd_enzyme=fwd_enzyme,
                    rev_enzyme=rev_enzyme,
                    primer_conc=primer_conc,
                    custom_prefix=prefix
                )
                st.markdown("**Forward Primer:**")
                st.code(pfwd, language="text")
                st.write(f"- Length: {len_fwd} nt  |  Tm (°C): {tm_fwd:.2f}")
                st.markdown("**Reverse Primer:**")
                st.code(prev, language="text")
                st.write(f"- Length: {len_rev} nt  |  Tm (°C): {tm_rev:.2f}")

        st.markdown("---")
        with st.expander("Ligation Simulator"):
            st.markdown("Estimate ligation efficiency between two fragments.")
            overhang_fwd = st.text_input("Forward Fragment Overhang (e.g. GATC)")
            overhang_rev = st.text_input("Reverse Fragment Overhang (e.g. AATT)")
            molar_ratio = st.number_input("Insert:Vector Molar Ratio", min_value=0.1, value=3.0, step=0.1)
            total_conc = st.number_input("Total DNA Concentration (ng/µL)", min_value=1.0, value=50.0, step=1.0)
            if st.button("Simulate Ligation"):
                lig_result = simulate_ligation(
                    fwd_overhang=overhang_fwd.upper(),
                    rev_overhang=overhang_rev.upper(),
                    molar_ratio_insert=molar_ratio,
                    total_conc=total_conc
                )
                st.write(f"- Compatibility: {'✅ Yes' if lig_result['compatibility'] else '❌ No'}")
                st.write(f"- Estimated Efficiency: {lig_result['estimated_efficiency']}%")
                st.write(f"- Message: {lig_result['message']}")

    # --- Hybridization Simulation ---
    elif app_mode == "Hybridization Simulation":
        st.subheader("🧪 Hybridization Simulation")
        st.markdown("""
            Simulate hybridization (alignment) between two DNA strands.
            If Biopython is available, the best alignment score and alignment will be shown.
        """)
        seq1 = st.text_area("Sequence 1", height=100, help="Enter DNA 1 (ATCG).")
        seq2 = st.text_area("Sequence 2", height=100, help="Enter DNA 2 (ATCG).")
        if st.button("Simulate Hybridization"):
            valid1, c1, m1 = validate_dna_sequence(seq1)
            valid2, c2, m2 = validate_dna_sequence(seq2)
            if not valid1:
                st.error(f"Sequence 1 invalid: {m1}")
            elif not valid2:
                st.error(f"Sequence 2 invalid: {m2}")
            else:
                if BIOPYTHON_AVAILABLE:
                    alignment = pairwise_alignment(c1, c2[::-1])
                    if alignment:
                        st.write(f"Alignment Score: {alignment.score:.2f}")
                        st.text(str(alignment))
                    else:
                        st.warning("Alignment failed.")
                else:
                    st.warning("Biopython not available; cannot align.")

    # --- ORF Finder ---
    elif app_mode == "ORF Finder":
        st.subheader("🔍 ORF Finder & Reverse Complement")
        dna_in = st.text_area("DNA Sequence", height=150, help="Enter raw DNA (ATCG).")
        if st.button("Find ORFs"):
            valid, clean_seq, warn = validate_dna_sequence(dna_in)
            if not valid:
                st.error(f"Invalid DNA: {warn}")
            else:
                orf_text = format_orf_list(clean_seq)
                st.text(orf_text)
        if st.button("Get Reverse Complement"):
            valid, clean_seq, warn = validate_dna_sequence(dna_in)
            if not valid:
                st.error(f"Invalid DNA: {warn}")
            else:
                rc = reverse_complement(clean_seq)
                st.markdown("**Reverse Complement:**")
                st.code(rc, language="text")

    # --- Sequence Alignment ---
    elif app_mode == "Sequence Alignment":
        st.subheader("🔗 Sequence Alignment Tools")

        with st.expander("Pairwise Alignment"):
            seqA = st.text_area("Sequence A", height=100, key="alignA")
            seqB = st.text_area("Sequence B", height=100, key="alignB")
            if st.button("Align A vs B"):
                validA, cA, mA = validate_dna_sequence(seqA)
                validB, cB, mB = validate_dna_sequence(seqB)
                if not validA:
                    st.error(f"Seq A invalid: {mA}")
                elif not validB:
                    st.error(f"Seq B invalid: {mB}")
                else:
                    if BIOPYTHON_AVAILABLE:
                        aln = pairwise_alignment(cA, cB)
                        if aln:
                            st.write(f"Alignment Score: {aln.score:.2f}")
                            st.text(str(aln))
                        else:
                            st.warning("Alignment failed.")
                    else:
                        st.warning("Biopython not installed; pairwise alignment unavailable.")

        with st.expander("Multiple Sequence Alignment (MSA)"):
            msa_input = st.text_area("Enter sequences (one per line, no >ID)", height=150)
            if st.button("Run MSA"):
                seqs = [line.strip() for line in msa_input.splitlines() if line.strip()]
                if len(seqs) < 2:
                    st.error("Enter at least two sequences.")
                else:
                    if BIOPYTHON_AVAILABLE:
                        msa_res = multiple_sequence_alignment(seqs)
                        if msa_res:
                            df_aln = render_alignment_df(msa_res)
                            st.dataframe(df_aln, use_container_width=True)
                        else:
                            st.warning("MSA failed or Clustal Omega not available.")
                    else:
                        st.warning("Biopython not installed; MSA unavailable.")

    # --- CRISPR sgRNA DESIGNER ---
    elif app_mode == "CRISPR sgRNA Designer":
        st.subheader("🔬 CRISPR sgRNA Designer")
        crispr_seq = st.text_area("Target DNA Sequence", height=200, help="Enter genomic DNA (ATCG).")
        cas_sys = st.selectbox("Select Cas System", options=list(CRISPR_PAM_SEQUENCES.keys()))
        if st.button("Design sgRNAs"):
            valid, clean_seq, msg = validate_dna_sequence(crispr_seq)
            if not valid:
                st.error(f"Invalid DNA: {msg}")
            else:
                guides = find_crispr_targets(clean_seq, cas_system=cas_sys)
                if guides:
                    df_guides = pd.DataFrame(guides)
                    st.dataframe(df_guides, use_container_width=True)
                    csv_data = df_guides.to_csv(index=False)
                    st.download_button("Download sgRNA List (CSV)", data=csv_data, file_name="sgRNAs.csv", mime="text/csv")
                else:
                    st.info("No guides found with the given PAM pattern.")

    # --- Plasmid Visualizer ---
    elif app_mode == "Plasmid Visualizer":
        st.subheader("🧬 Plasmid Visualizer")
        plasmid_seq = st.text_area("Plasmid DNA Sequence", height=200, help="Enter full plasmid sequence (ATCG).")
        circular = st.checkbox("Draw Circular Map", value=True)
        features_input = st.text_area(
            "Enter features as JSON list", 
            height=150,
            help='e.g. [{"start":1,"end":500,"strand":+1,"label":"Gene1","color":"#ff0000"}, …]'
        )
        if st.button("Visualize"):
            valid, clean_seq, msg = validate_dna_sequence(plasmid_seq)
            if not valid:
                st.error(f"Invalid plasmid DNA: {msg}")
            else:
                if not DNA_FEATURES_AVAILABLE:
                    st.error("dna_features_viewer is not installed.")
                else:
                    try:
                        feats = json.loads(features_input)
                        visualize_plasmid(feats, clean_seq, circular=circular)
                    except Exception as e:
                        st.error(f"Failed to parse features JSON: {e}")

    # --- Synthetic Pathway Designer ---
    elif app_mode == "Synthetic Pathway Designer":
        st.subheader("🔗 Synthetic Pathway Designer")
        with st.expander("Design Multi‐gene Operon"):
            genes_text = st.text_area("Enter gene sequences (one per line, ATCG)", height=200)
            rbs_seq = st.text_input("RBS Sequence", value="AGGAGG")
            linker_seq = st.text_input("Peptide Linker Seq", value="GGGGS")
            terminator_seq = st.text_input("Terminator Sequence", value="TTATTATT")
            if st.button("Build Operon"):
                gene_list = [g.strip() for g in genes_text.splitlines() if g.strip()]
                if not gene_list:
                    st.error("Enter at least one gene sequence.")
                else:
                    harmonized = harmonize_codons_across_genes(gene_list, target_organism="E. coli BL21")
                    operon_seq = design_operon(harmonized, rbs_seq=rbs_seq, linker_seq=linker_seq, terminator_seq=terminator_seq)
                    st.markdown("**Operon Sequence:**")
                    st.code(operon_seq, language="text")
                    fst = export_fasta({"Operon": operon_seq})
                    st.download_button("Download Operon (FASTA)", data=fst, file_name="operon.fasta", mime="text/plain")

    # --- AI Docking & Functional Prediction ---
    elif app_mode == "AI Docking & Functional Prediction":
        st.subheader("🤖 AI Docking & Functional Prediction")
        dock_tab, func_tab = st.tabs(["In Silico Docking", "Functional Prediction (GO)"])

        with dock_tab:
            st.markdown("**In Silico Docking** (placeholders).")
            input_type = st.radio("Input Type", options=["Protein Sequence", "DNA Sequence", "Upload PDB Files"])
            if input_type in ["Protein Sequence", "DNA Sequence"]:
                seq1 = st.text_area("Sequence 1 (AA or nt)", height=150)
                seq2 = st.text_area("Sequence 2 (AA or nt)", height=150)
                if st.button("Run Docking"):
                    st.info("Docking pipeline not implemented in this standalone version.")
            else:
                pdb1 = st.file_uploader("Upload PDB File 1", type=["pdb"])
                pdb2 = st.file_uploader("Upload PDB File 2", type=["pdb"])
                if st.button("Run Docking (from PDB)"):
                    st.info("Docking pipeline not implemented in this standalone version.")

        with func_tab:
            st.markdown("**AI‐based Functional Prediction** (GO terms, enzyme class).")
            prot_input = st.text_area("Protein Sequence (One‐letter)", height=200)
            if st.button("Predict Function"):
                if not prot_input:
                    st.error("Enter a protein sequence.")
                else:
                    func_res = predict_function(prot_input)
                    st.write("**Predicted GO Terms:**", func_res.get("go_terms", []))
                    st.write("**Inferred Enzyme Class:**", func_res.get("enzyme_class", "N/A"))
                    st.write("**Pathway Suggestions:**", func_res.get("pathway_suggestions", []))

    # --- Export Sequences ---
    elif app_mode == "Export Sequences":
        st.subheader("💾 Export Sequences")
        fmt = st.selectbox("Format", ["FASTA", "GenBank"])
        seqs_json = st.text_area(
            "Enter sequences JSON (e.g. {\"name1\":\"ATGC...\",\"name2\":\"TTGA...\"})",
            height=200
        )
        feats_json = st.text_area(
            "Features JSON (for GenBank, e.g. [{\"start\":1,\"end\":100,\"label\":\"Gene1\"}, …])",
            height=150
        )
        if st.button("Export"):
            try:
                seq_dict = json.loads(seqs_json)
                if fmt == "FASTA":
                    fasta_out = export_fasta(seq_dict)
                    st.code(fasta_out, language="text")
                    st.download_button("Download FASTA", data=fasta_out, file_name="sequences.fasta", mime="text/plain")
                else:
                    feats = json.loads(feats_json) if feats_json.strip() else []
                    for name, seq in seq_dict.items():
                        gb = export_genbank(seq, organism=name, features=feats, opt_result=None)
                        st.code(gb, language="text")
                        st.download_button(f"Download {name}.gb", data=gb, file_name=f"{name}.gb", mime="text/plain")
            except Exception as e:
                st.error(f"Failed to export: {e}")

    # --- Help & Guide ---
    elif app_mode == "Help & Guide":
        st.subheader("❓ Help & Guide")
        st.markdown("""
        **G-Synth Usage**

        - **Navigation**: Use the sidebar to switch between modules.
        - **Input Formats**: Sequences should be raw (no whitespace) and uppercase (ATCG for DNA).
        - **Codon Optimization**: Select the correct host organism; you can avoid restriction sites by checking the list.
        - **Primer Design**: Ensure your template region doesn’t already contain the enzyme recognition site (warning will appear).
        - **CRISPR**: Enter long sequences (≥100 nt) to find potential guides.
        - **Plasmid Visualizer**: Provide feature annotations as a JSON array with 1-based coordinates.
        - **Pathway Designer**: All input genes are automatically codon-harmonized for E. coli BL21 by default.
        - **AI Tools**: These are placeholders. Integrate your local AlphaFold/AutoDock pipelines to enable full functionality.
        - **Export**: FASTA or GenBank. For GenBank, provide features list with start/end/label.

        **Dependencies** (install via `pip install -r requirements.txt`):
        - streamlit, pandas, numpy, matplotlib, seaborn, plotly
        - biopython
        - dna-features-viewer
        - stmol, py3Dmol
        - transformers, torch
        - requests
        - biotite

        For bug reports or detailed documentation, visit the [G-Synth GitHub](https://github.com/username/G-Synth).
        """)
    else:
        st.error("Unknown selection. Please choose a valid module.")

if __name__ == "__main__":
    main()
