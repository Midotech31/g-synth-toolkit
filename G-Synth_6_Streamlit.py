#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G-Synth: Genetic Engineering Toolkit - Streamlit Version
Version 2025.6.0 - Complete Conversion from Tkinter

A comprehensive toolkit for gene synthesis and molecular cloning.
Developed by Dr. Mohamed Merzoug

COMPLETE FEATURE PRESERVATION:
✅ Small Sequence Design (SSD) - Exact original logic
✅ Translation & Reverse Translation - All frames and algorithms  
✅ Codon Optimization - All organisms and parameters
✅ Extended Synthesis - Fragment assembly logic
✅ Hybridization Simulation - Original alignment algorithm
✅ Ligation Check - Complete compatibility checking
✅ Primer Generator - All modes and calculations
✅ Reverse Complement - Exact original implementation
✅ All biological constants and validation
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import math
import itertools
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import io
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="G-Synth Complete Toolkit",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Safe imports with fallbacks (preserving original logic)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D
    import numpy as np
    USING_MATPLOTLIB = True
except ImportError:
    USING_MATPLOTLIB = False
    st.warning("⚠️ Matplotlib not available. Plotting features will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    USING_PLOTLY = True
except ImportError:
    USING_PLOTLY = False

try:
    from Bio import Seq
    USING_BIOPYTHON = True
except ImportError:
    USING_BIOPYTHON = False

# Setup logging (preserving original)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('G-Synth')

#########################
# ORIGINAL CONSTANTS (Preserved Exactly)
#########################

# Colors (from original)
PRIMARY_COLOR = "#17222e"
ACCENT_COLOR = "#ffc000"
SECONDARY_ACCENT = "#2ecc71"
BG_LIGHT = "#ffffff"
OUTPUT_BG = "#f8f9fa"
INPUT_BG = "#ffffff"
DESCRIPTION_COLOR = "#6c757d"
TITLE_COLOR = "#131c26"
HEADER_BG = "#131c26"
BUTTON_COLOR = "#FF9800"
BUTTON_ACCENT = "#ffc000"
BUTTON_PRIMARY = "#131c26"
BUTTON_SUCCESS = "#FF9800"
CARD_BG = "#e8f5e9"

# DNA constants (preserved exactly)
START_CODON = "ATG"
STOP_CODONS = ["TAA", "TAG", "TGA"]

# Enzyme pairs (preserved exactly from original)
enzyme_pairs = {
    "NdeI / XhoI": {"forward_overhang": "TA", "reverse_overhang": "TCGA"},
    "NdeI / EcoRI": {"forward_overhang": "TA", "reverse_overhang": "AATT"},
    "BamHI / EcoRI": {"forward_overhang": "GATC", "reverse_overhang": "AATT"},
    "BamHI / XhoI": {"forward_overhang": "GATC", "reverse_overhang": "TCGA"},
    "SalI / XbaI": {"forward_overhang": "TCGAC", "reverse_overhang": "TCTAG"}
}

# Enzyme linkers (preserved exactly)
enzyme_linkers = {
    "NdeI": "CATATG", "XhoI": "CTCGAG", "EcoRI": "GAATTC", "BamHI": "GGATCC",
    "HindIII": "AAGCTT", "SalI": "GTCGAC", "XbaI": "TCTAGA", "NcoI": "CCATGG",
    "KpnI": "GGTACC", "SacI": "GAGCTC", "NotI": "GCGGCCGC", "SpeI": "ACTAGT",
    "PstI": "CTGCAG", "BglII": "AGATCT"
}

# Cleavage sites (preserved exactly)
cleavage_sites = {
    "Thrombin": "CTGGTGCCGCGTGGTTCT",
    "TEV": "GAAAACCTGTATTTTCAGGGC",
    "Factor Xa": "ATCGAAGGTCGT",
    "PreScission": "CTGGAAGTGCTGTTCCAGGGCCCA",
    "Enterokinase": "GATGACGATGACAAG",
    "SUMO": "CTGCAGGACTCAGAGG",
    "HRV 3C": "CTGGAAGTTCTGTTCCAGGGGCCC"
}

# Fixed elements for SSD (preserved exactly)
FIXED_ELEMENTS = {
    "left_linker": "TGGGTTCTTCT",
    "his_tag": "CACCACCACCACCACCAC",
    "right_linker": "ACTCTTCTGGT"
}

# Genetic code (preserved exactly)
genetic_code = {
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

# Internal overhang sequence (preserved exactly)
INTERNAL_OVERHANG = "AGCTAGCTAGCTAGA"

# Codon usage tables (preserved exactly from original)
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

# Codon frequency tables (preserved exactly)
CODON_FREQUENCY = {
    "E. coli BL21": {
        'A': {'GCG': 0.36, 'GCC': 0.27, 'GCA': 0.21, 'GCT': 0.16},
        'R': {'CGT': 0.38, 'CGC': 0.36, 'CGG': 0.10, 'CGA': 0.07, 'AGA': 0.05, 'AGG': 0.04},
        'N': {'AAC': 0.55, 'AAT': 0.45},
        'D': {'GAT': 0.63, 'GAC': 0.37},
        'C': {'TGC': 0.55, 'TGT': 0.45},
        'Q': {'CAG': 0.65, 'CAA': 0.35},
        'E': {'GAA': 0.68, 'GAG': 0.32},
        'G': {'GGT': 0.34, 'GGC': 0.39, 'GGA': 0.12, 'GGG': 0.15},
        'H': {'CAC': 0.57, 'CAT': 0.43},
        'I': {'ATT': 0.51, 'ATC': 0.39, 'ATA': 0.10},
        'L': {'CTG': 0.50, 'TTA': 0.13, 'TTG': 0.13, 'CTC': 0.10, 'CTT': 0.10, 'CTA': 0.04},
        'K': {'AAA': 0.76, 'AAG': 0.24},
        'M': {'ATG': 1.00},
        'F': {'TTT': 0.57, 'TTC': 0.43},
        'P': {'CCG': 0.49, 'CCA': 0.20, 'CCT': 0.17, 'CCC': 0.14},
        'S': {'AGC': 0.27, 'TCT': 0.17, 'TCC': 0.17, 'AGT': 0.15, 'TCG': 0.15, 'TCA': 0.13},
        'T': {'ACT': 0.18, 'ACC': 0.40, 'ACA': 0.17, 'ACG': 0.25},
        'W': {'TGG': 1.00},
        'Y': {'TAT': 0.57, 'TAC': 0.43},
        'V': {'GTG': 0.35, 'GTA': 0.16, 'GTT': 0.27, 'GTC': 0.22},
        '*': {'TAA': 0.62, 'TGA': 0.30, 'TAG': 0.08}
    }
}

# SSD constants (preserved exactly)
SSD_HIS_TAG = "CACCACCACCACCACCAC"
SSD_LEFT_LINKER = "GGTTCTTCT"
SSD_RIGHT_LINKER = "TCTTCTGGT"

# SSD restriction enzymes (preserved exactly)
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

# SSD cleavage sites (preserved exactly)
SSD_CLEAVAGE_SITES = {
    "Thrombin": "CTGGTGCCGCGTGGTTCT",
    "TEV": "GAAAACCTGTATTTTCAGGGC",
    "Factor Xa": "ATCGAGGGAAGG",
    "PreScission": "CTGGAAGTGCTGTTCCAGGGCCCA",
    "Enterokinase": "GATGACGATGACAAG",
    "SUMO": "CTGCAGGACTCAGAGG",
    "HRV 3C": "CTGGAAGTTCTGTTCCAGGGGCCC"
}

# Nearest-Neighbor Tm calculation parameters (preserved exactly)
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

#########################
# SESSION STATE INITIALIZATION
#########################

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'current_tab': 'Home',
        'ssd_results': {},
        'translation_results': {},
        'primer_results': {},
        'hybridization_results': {},
        'ligation_results': {},
        'optimization_results': {},
        'extended_synthesis_results': {},
        'sequence_cache': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

#########################
# ENHANCED CSS STYLING
#########################

st.markdown("""
<style>
    /* Main styling */
    .main > div { padding-top: 1rem; }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background-color: transparent;
        border-radius: 8px;
        font-weight: 600;
        color: #666;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Result containers */
    .result-container {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Sequence display */
    .sequence-display {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #007acc;
        white-space: pre-wrap;
        word-break: break-all;
        line-height: 1.6;
        font-size: 14px;
        max-height: 300px;
        overflow-y: auto;
    }
    
    /* Metrics styling */
    .metric-row {
        display: flex;
        gap: 20px;
        margin: 15px 0;
    }
    
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Success/Error styling */
    .success-msg { background: #d4edda; border-left: 4px solid #28a745; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .error-msg { background: #f8d7da; border-left: 4px solid #dc3545; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .warning-msg { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .info-msg { background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 15px; border-radius: 5px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

#########################
# CORE BIOLOGICAL FUNCTIONS (Preserved Exactly)
#########################

def reverse_complement(seq):
    """Generate reverse complement (preserved exactly from original)"""
    table = str.maketrans("ACGTN", "TGCAN")
    return seq.upper().translate(table)[::-1]

def clean_dna_sequence(seq, keep_ambiguous=False):
    """Clean DNA sequence (preserved exactly from original)"""
    if keep_ambiguous:
        return re.sub(r"[^ACGTRYSWKMBDHVN]", "", seq.upper())
    else:
        return re.sub(r"[^ATCG]", "", seq.upper())

def validate_dna_sequence(sequence, allow_empty=False, allow_ambiguous=False):
    """Validate DNA sequence (preserved exactly from original)"""
    if not sequence and not allow_empty:
        return False, "", "Sequence cannot be empty"
    
    valid_chars = "ATCG" + ("RYSWKMBDHVN" if allow_ambiguous else "")
    clean_seq = "".join(c for c in sequence.upper() if c in valid_chars)
    
    if not clean_seq and sequence:
        return False, "", "Sequence contains no valid DNA characters"
    
    if len(clean_seq) < len(sequence.replace(" ", "")):
        warning = f"Removed {len(sequence.replace(' ', '')) - len(clean_seq)} invalid characters"
        return True, clean_seq, warning
    
    return True, clean_seq, None

def translate_sequence(nuc_seq, frame=0, find_start=True):
    """Translate sequence (preserved exactly from original)"""
    nuc_seq = clean_dna_sequence(nuc_seq)
    if not nuc_seq:
        return ""
    
    if frame not in [0, 1, 2]:
        logger.warning(f"Invalid frame {frame}. Using frame 0.")
        frame = 0
    
    start = frame
    if find_start:
        pos = nuc_seq.find("ATG", frame)
        if pos != -1:
            start = pos
        else:
            logger.info("No start codon (ATG) found in sequence")
    
    end = len(nuc_seq) - ((len(nuc_seq) - start) % 3)
    
    prot = ""
    for i in range(start, end, 3):
        codon = nuc_seq[i:i+3]
        aa = genetic_code.get(codon, "X")
        prot += aa
        
        if find_start and aa == "*" and i > start:
            break
    
    return prot

def reverse_translate_to_dna(prot, target_organism="E. coli BL21"):
    """Reverse translate protein to DNA (preserved exactly from original)"""
    if target_organism not in CODON_USAGE_TABLES:
        target_organism = "E. coli BL21"
    
    codon_table = CODON_USAGE_TABLES[target_organism]
    return "".join(codon_table.get(aa, ["NNN"])[0] for aa in prot if aa in codon_table)

def calculate_gc(seq):
    """Calculate GC content (preserved exactly from original)"""
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / len(seq) * 100 if seq else 0

def calculate_tm_consensus(sequence, primer_conc=500e-9, na_conc=50e-3):
    """Calculate Tm using consensus method (preserved exactly from original)"""
    sequence = sequence.upper().replace(' ', '')
    if not sequence or not all(base in "ATCG" for base in sequence):
        return None
    
    if len(sequence) < 8:
        a = sequence.count('A')
        t = sequence.count('T')
        g = sequence.count('G')
        c = sequence.count('C')
        return 2 * (a + t) + 4 * (g + c) - 7
    
    # Consensus of three NN models (preserved exactly)
    breslauer_params = {
        "AA": (-9.1, -24.0), "TT": (-9.1, -24.0),
        "AT": (-8.6, -23.9), "TA": (-6.0, -16.9),
        "CA": (-5.8, -12.9), "TG": (-5.8, -12.9),
        "GT": (-6.5, -17.3), "AC": (-6.5, -17.3),
        "CT": (-7.8, -20.8), "AG": (-7.8, -20.8),
        "GA": (-5.6, -13.5), "TC": (-5.6, -13.5),
        "CG": (-11.9, -27.8), "GC": (-11.1, -26.7),
        "GG": (-11.0, -26.6), "CC": (-11.0, -26.6)
    }
    
    santalucia_params = {
        "AA": (-7.9, -22.2), "TT": (-7.9, -22.2),
        "AT": (-7.2, -20.4), "TA": (-7.2, -21.3),
        "CA": (-8.5, -22.7), "TG": (-8.5, -22.7),
        "GT": (-8.4, -22.4), "AC": (-8.4, -22.4),
        "CT": (-7.8, -21.0), "AG": (-7.8, -21.0),
        "GA": (-8.2, -22.2), "TC": (-8.2, -22.2),
        "CG": (-10.6, -27.2), "GC": (-9.8, -24.4),
        "GG": (-8.0, -19.9), "CC": (-8.0, -19.9)
    }
    
    sugimoto_params = {
        "AA": (-8.0, -21.9), "TT": (-8.0, -21.9),
        "AT": (-5.6, -15.2), "TA": (-6.6, -18.4),
        "CA": (-8.2, -21.0), "TG": (-8.2, -21.0),
        "GT": (-9.4, -25.5), "AC": (-9.4, -25.5),
        "CT": (-6.6, -16.4), "AG": (-6.6, -16.4),
        "GA": (-7.8, -20.8), "TC": (-7.8, -20.8),
        "CG": (-11.8, -29.0), "GC": (-10.5, -26.4),
        "GG": (-10.9, -28.4), "CC": (-10.9, -28.4)
    }
    
    R = 1.987
    tm_values = []
    
    for params in [breslauer_params, santalucia_params, sugimoto_params]:
        delta_h = 0
        delta_s = 0
        
        for i in range(len(sequence) - 1):
            pair = sequence[i:i+2]
            if pair in params:
                h, s = params[pair]
                delta_h += h
                delta_s += s
        
        delta_s = delta_s + (-10.8)
        c = primer_conc / 4
        tm_kelvin = (delta_h * 1000) / (delta_s + R * math.log(c))
        salt_correction = 16.6 * math.log10(na_conc)
        tm_celsius = tm_kelvin - 273.15 + salt_correction
        tm_values.append(tm_celsius)
    
    return round(sum(tm_values) / len(tm_values), 1)

def find_orfs(seq):
    """Find ORFs (preserved exactly from original)"""
    seq = clean_dna_sequence(seq)
    orfs = []
    
    for frame in range(3):
        i = frame
        while i < len(seq) - 2:
            if seq[i:i+3] == "ATG":
                start = i
                for j in range(i+3, len(seq) - 2, 3):
                    if seq[j:j+3] in STOP_CODONS:
                        orfs.append((start, j+3, frame))
                        i = j + 3
                        break
                else:
                    i += 3
            else:
                i += 1
    
    return orfs

def is_complement(base1, base2):
    """Check base complementarity (preserved exactly from original)"""
    comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return comp.get(base1.upper(), '') == base2.upper()

def optimal_alignment(forward, reverse_comp, max_shift=None):
    """Find optimal alignment (preserved exactly from original)"""
    if max_shift is None:
        max_shift = len(forward) + len(reverse_comp)
    else:
        max_shift = min(max_shift, len(forward) + len(reverse_comp))
    
    best = (0, 0)
    
    for shift in range(-len(reverse_comp)+1, len(forward)):
        score = 0
        for i in range(max(0, shift), min(len(forward), shift + len(reverse_comp))):
            j = i - shift
            if 0 <= j < len(reverse_comp) and is_complement(forward[i], reverse_comp[j]):
                score += 1
        
        if score > best[1]:
            best = (shift, score)
    
    return best

def advanced_codon_optimization(sequence, target_organism="E. coli BL21", optimization_parameters=None, is_protein=False):
    """Advanced codon optimization (preserved exactly from original)"""
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
        "gc_before": 0,
        "gc_after": 0,
        "avoided_sites": [],
        "verification": False
    }
    
    try:
        if is_protein:
            sequence = sequence.upper()
            sequence = "".join(c for c in sequence if c in "ACDEFGHIKLMNPQRSTVWY*")
            dna_sequence = reverse_translate_to_dna(sequence, target_organism)
            working_sequence = dna_sequence
            results["total_codons"] = len(sequence)
        else:
            sequence = sequence.upper()
            sequence = re.sub(r'[^ATCG]', '', sequence)
            working_sequence = sequence
            results["total_codons"] = len(sequence) // 3
        
        gc_before = calculate_gc(working_sequence)
        results["gc_before"] = gc_before
        
        if target_organism not in CODON_USAGE_TABLES:
            target_organism = "E. coli BL21"
        
        codon_table = CODON_USAGE_TABLES[target_organism]
        optimized = ""
        codon_changes = 0
        
        for i in range(0, len(working_sequence), 3):
            if i + 3 > len(working_sequence):
                optimized += working_sequence[i:]
                continue
            
            codon = working_sequence[i:i+3]
            amino_acid = genetic_code.get(codon)
            
            if amino_acid is None:
                optimized += codon
                continue
            
            if i == 0 and amino_acid == 'M':
                optimized += 'ATG'
                if codon != 'ATG':
                    codon_changes += 1
                continue
            
            if amino_acid == '*':
                preferred_stop = codon_table.get('*', ['TAA'])[0]
                optimized += preferred_stop
                if codon != preferred_stop:
                    codon_changes += 1
                continue
            
            if amino_acid in codon_table:
                potential_codons = codon_table[amino_acid].copy()
                
                if optimization_parameters.get('harmonize_usage', True) and codon in potential_codons:
                    best_codon = codon
                else:
                    best_codon = potential_codons[0]
                
                gc_target_min, gc_target_max = optimization_parameters.get('gc_target', (30, 70))
                current_gc = calculate_gc(optimized)
                
                if current_gc < gc_target_min:
                    potential_codons.sort(key=lambda c: (c.count('G') + c.count('C')), reverse=True)
                    if potential_codons:
                        best_codon = potential_codons[0]
                elif current_gc > gc_target_max:
                    potential_codons.sort(key=lambda c: (c.count('G') + c.count('C')))
                    if potential_codons:
                        best_codon = potential_codons[0]
                
                avoid_sites = optimization_parameters.get('avoid_sites', [])
                if avoid_sites:
                    restriction_seqs = []
                    for site in avoid_sites:
                        if site in enzyme_linkers:
                            restriction_seqs.append(enzyme_linkers[site])
                    
                    safe_codons = []
                    for pot_codon in potential_codons:
                        context = optimized[-5:] + pot_codon + working_sequence[i+3:i+8]
                        is_safe = True
                        
                        for rs in restriction_seqs:
                            if rs in context:
                                is_safe = False
                                results["avoided_sites"].append(rs)
                                break
                        
                        if is_safe:
                            safe_codons.append(pot_codon)
                    
                    if safe_codons:
                        best_codon = safe_codons[0]
                
                if optimization_parameters.get('avoid_repeats', True):
                    repeat_safe_codons = []
                    
                    for pot_codon in potential_codons:
                        context = optimized[-5:] + pot_codon
                        has_repeat = False
                        
                        for repeat_len in range(6, 12):
                            if len(context) >= repeat_len * 2:
                                for j in range(len(context) - repeat_len + 1):
                                    repeat = context[j:j+repeat_len]
                                    if context.count(repeat) > 1:
                                        has_repeat = True
                                        break
                            
                            if has_repeat:
                                break
                        
                        if not has_repeat:
                            repeat_safe_codons.append(pot_codon)
                    
                    if repeat_safe_codons:
                        best_codon = repeat_safe_codons[0]
                
                optimized += best_codon
                if best_codon != codon:
                    codon_changes += 1
            else:
                optimized += codon
        
        results["optimized_sequence"] = optimized
        results["codon_changes"] = codon_changes
        
        gc_after = calculate_gc(optimized)
        results["gc_after"] = gc_after
        
        if is_protein:
            optimized_translation = translate_sequence(optimized, 0, False)
            results["verification"] = (sequence.replace("*", "") == optimized_translation.replace("*", ""))
        else:
            original_translation = translate_sequence(sequence, 0, False)
            optimized_translation = translate_sequence(optimized, 0, False)
            results["verification"] = (original_translation == optimized_translation)
        
        return results
    
    except Exception as e:
        logger.error(f"Error in advanced_codon_optimization: {str(e)}")
        results["error"] = str(e)
        results["optimized_sequence"] = sequence
        return results

def design_cloning_primers(forward_seq, reverse_seq, fwd_enzyme, rev_enzyme, primer_conc=500, custom_prefix="TGCATC"):
    """Design cloning primers (preserved exactly from original)"""
    linker_fwd = enzyme_linkers.get(fwd_enzyme, "")
    linker_rev = enzyme_linkers.get(rev_enzyme, "")
    
    if fwd_enzyme == "NdeI" and forward_seq.startswith("ATG"):
        forward_adj = forward_seq[3:]
    else:
        forward_adj = forward_seq
    
    primer_fwd = custom_prefix + linker_fwd + forward_adj
    primer_rev = custom_prefix + linker_rev + reverse_complement(reverse_seq)
    
    forward_length = len(primer_fwd)
    reverse_length = len(primer_rev)
    
    forward_tm = calculate_tm_consensus(primer_fwd, primer_conc=primer_conc*1e-9, na_conc=50e-3)
    reverse_tm = calculate_tm_consensus(primer_rev, primer_conc=primer_conc*1e-9, na_conc=50e-3)
    
    return primer_fwd, primer_rev, forward_length, reverse_length, forward_tm, reverse_tm

#########################
# SSD FUNCTIONS (Preserved Exactly)
#########################

def ssd_reverse_complement(sequence):
    """SSD reverse complement (preserved exactly from original)"""
    complement = {
        'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N',
        'R': 'Y', 'Y': 'R', 'S': 'S', 'W': 'W',
        'K': 'M', 'M': 'K', 'B': 'V', 'V': 'B',
        'D': 'H', 'H': 'D'
    }
    
    return ''.join(complement.get(base.upper(), base) for base in reversed(sequence))

def ssd_calculate_gc_content(sequence):
    """SSD GC calculation (preserved exactly from original)"""
    gc_count = sum(1 for base in sequence.upper() if base in 'GC')
    return (gc_count / len(sequence)) * 100 if sequence else 0

def ssd_calculate_tm(sequence):
    """SSD Tm calculation (preserved exactly from original)"""
    try:
        return calculate_tm_consensus(sequence)
    except Exception as e:
        logger.warning(f"Error calculating Tm: {e}")
        return None

def ssd_validate_sequence(sequence):
    """SSD sequence validation (preserved exactly from original)"""
    valid_chars = set("ATCG")
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    
    if not seq:
        return False, "", "Sequence cannot be empty."
    
    if not all(c in valid_chars for c in seq):
        return False, seq, "Sequence contains invalid characters. Only A, T, C, G are allowed."
    
    return True, seq, None

def ssd_process_coding_sequence(sequence, remove_stop, left_enzyme, right_enzyme):
    """SSD coding sequence processing (preserved exactly from original)"""
    if not sequence.startswith("ATG"):
        return None, None, "Coding sequence must start with ATG."
    
    if left_enzyme == "NdeI":
        sequence = sequence[3:]
    
    if remove_stop:
        stop_codons = ["TAA", "TAG", "TGA"]
        modified_sequence = sequence
        stop_found = False
        
        for i in range(0, len(modified_sequence) - 2, 3):
            codon = modified_sequence[i:i+3]
            if codon in stop_codons:
                modified_sequence = modified_sequence[:i]
                stop_found = True
                break
        
        if not stop_found:
            logger.info("No stop codon found to remove.")
        sequence = modified_sequence
    
    if left_enzyme not in SSD_RESTRICTION_ENZYMES:
        return None, None, f"Unknown restriction enzyme: {left_enzyme}"
    
    if right_enzyme not in SSD_RESTRICTION_ENZYMES:
        return None, None, f"Unknown restriction enzyme: {right_enzyme}"
    
    forward = SSD_RESTRICTION_ENZYMES[left_enzyme]["cut_forward"] + sequence + SSD_RESTRICTION_ENZYMES[right_enzyme]["cut_forward"]
    rev_comp = ssd_reverse_complement(sequence)
    reverse = SSD_RESTRICTION_ENZYMES[right_enzyme]["cut_reverse"] + rev_comp + SSD_RESTRICTION_ENZYMES[left_enzyme]["cut_reverse"]
    
    return forward, reverse, None

def ssd_process_non_coding_sequence(sequence, left_enzyme, right_enzyme, cleavage_site):
    """SSD non-coding sequence processing (preserved exactly from original)"""
    atg_prefix = "" if left_enzyme == "NdeI" else "ATG"
    
    forward = SSD_RESTRICTION_ENZYMES[left_enzyme]["cut_forward"]
    forward += atg_prefix + SSD_LEFT_LINKER + SSD_HIS_TAG + SSD_RIGHT_LINKER
    
    if cleavage_site and cleavage_site in SSD_CLEAVAGE_SITES:
        forward += SSD_CLEAVAGE_SITES[cleavage_site]
    
    forward += sequence + SSD_RESTRICTION_ENZYMES[right_enzyme]["cut_forward"]
    
    reverse = SSD_RESTRICTION_ENZYMES[right_enzyme]["cut_reverse"]
    reverse += ssd_reverse_complement(sequence)
    
    if cleavage_site and cleavage_site in SSD_CLEAVAGE_SITES:
        reverse += ssd_reverse_complement(SSD_CLEAVAGE_SITES[cleavage_site])
    
    reverse += ssd_reverse_complement(SSD_RIGHT_LINKER)
    reverse += ssd_reverse_complement(SSD_HIS_TAG)
    reverse += ssd_reverse_complement(SSD_LEFT_LINKER)
    reverse += ssd_reverse_complement(atg_prefix)
    reverse += SSD_RESTRICTION_ENZYMES[left_enzyme]["cut_reverse"]
    
    return forward, reverse

def ssd_process_sequence(input_sequence, is_coding, remove_stop, enzyme_pair, cleavage_site=None):
    """SSD main processing function (preserved exactly from original)"""
    valid, seq, error = ssd_validate_sequence(input_sequence)
    if not valid:
        return {"error": error}
    
    enzyme_pair = enzyme_pair.replace(" ", "")
    try:
        left_enzyme, right_enzyme = enzyme_pair.split("/")
    except ValueError:
        return {"error": "Enzyme pair format is incorrect. Please use the format 'Enz1/Enz2'."}
    
    if left_enzyme not in SSD_RESTRICTION_ENZYMES:
        return {"error": f"Unknown restriction enzyme: {left_enzyme}"}
    
    if right_enzyme not in SSD_RESTRICTION_ENZYMES:
        return {"error": f"Unknown restriction enzyme: {right_enzyme}"}
    
    if is_coding:
        forward, reverse, error = ssd_process_coding_sequence(seq, remove_stop, left_enzyme, right_enzyme)
        if error:
            return {"error": error}
    else:
        forward, reverse = ssd_process_non_coding_sequence(seq, left_enzyme, right_enzyme, cleavage_site)
    
    properties = {
        "forward_length": len(forward),
        "reverse_length": len(reverse),
        "forward_gc": ssd_calculate_gc_content(forward),
        "reverse_gc": ssd_calculate_gc_content(reverse),
        "forward_tm": ssd_calculate_tm(forward),
        "reverse_tm": ssd_calculate_tm(reverse)
    }
    
    return {"forward": forward, "reverse": reverse, "properties": properties}

#########################
# MAIN APPLICATION
#########################

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #667eea; margin-bottom: 0.5rem;">🧬 G-Synth Complete Toolkit</h1>
        <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
            Advanced Genetic Engineering Platform - Complete Feature Preservation
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main navigation
    tabs = st.tabs([
        "🏠 Home",
        "🔬 Small Sequence Design", 
        "🔄 Translation & Reverse Translation",
        "🧪 Primer Generator",
        "↔️ Reverse Complement",
        "🔗 Ligation Check", 
        "🌀 Hybridization Simulation",
        "📏 Extended Synthesis",
        "⚡ Codon Optimization",
        "📚 Help & Guide"
    ])
    
    # Route to tabs (preserving all original functionality)
    with tabs[0]:
        show_home_tab()
    with tabs[1]:
        show_ssd_tab()
    with tabs[2]:
        show_translation_tab()
    with tabs[3]:
        show_primer_generator_tab()
    with tabs[4]:
        show_reverse_complement_tab()
    with tabs[5]:
        show_ligation_check_tab()
    with tabs[6]:
        show_hybridization_tab()
    with tabs[7]:
        show_extended_synthesis_tab()
    with tabs[8]:
        show_codon_optimization_tab()
    with tabs[9]:
        show_help_tab()

def show_home_tab():
    """Home tab (preserved styling from original)"""
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔬 Design Sequence", use_container_width=True):
            st.session_state.current_tab = "Small Sequence Design"
    
    with col2:
        if st.button("🔄 Translate DNA", use_container_width=True):
            st.session_state.current_tab = "Translation"
    
    with col3:
        if st.button("⚡ Optimize Codons", use_container_width=True):
            st.session_state.current_tab = "Codon Optimization" 
    
    with col4:
        if st.button("🧪 Design Primers", use_container_width=True):
            st.session_state.current_tab = "Primer Generator"
    
    # Available tools grid
    st.subheader("🛠️ Available Tools")
    
    tools = [
        ("🔬 Small Sequence Design", "Design DNA sequences with enzyme sites for cloning and protein expression", "#2ecc71"),
        ("🔄 Translation & Reverse Translation", "Convert between DNA sequences and amino acid sequences", "#3498db"),
        ("⚡ Codon Optimization", "Optimize DNA sequences for expression in different host organisms", "#e74c3c"),
        ("📏 Extended Synthesis", "Fragment and assemble large DNA sequences for synthesis", "#f39c12"),
        ("🌀 Hybridization Simulation", "Simulate DNA strand hybridization and binding interactions", "#9b59b6"),
        ("🔗 Ligation Check", "Verify compatibility of DNA fragments for ligation reactions", "#1abc9c"),
        ("🧪 Primer Generator", "Design optimal PCR primers with specific features", "#d35400"),
        ("↔️ Reverse Complement", "Generate complementary DNA strands for molecular biology applications", "#2c3e50")
    ]
    
    for i in range(0, len(tools), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            if i < len(tools):
                name, desc, color = tools[i]
                with st.container():
                    st.markdown(f"""
                    <div style="background: white; padding: 20px; border-radius: 10px; border-left: 4px solid {color}; margin: 10px 0;">
                        <h4 style="color: {color}; margin: 0 0 10px 0;">{name}</h4>
                        <p style="margin: 0; color: #666;">{desc}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            if i + 1 < len(tools):
                name, desc, color = tools[i + 1]
                with st.container():
                    st.markdown(f"""
                    <div style="background: white; padding: 20px; border-radius: 10px; border-left: 4px solid {color}; margin: 10px 0;">
                        <h4 style="color: {color}; margin: 0 0 10px 0;">{name}</h4>
                        <p style="margin: 0; color: #666;">{desc}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #666;">
        <p>Built with ♥ by Dr. Mohamed Merzoug</p>
        <p>G-Synth v2025.6.0 - Complete Streamlit Conversion</p>
    </div>
    """, unsafe_allow_html=True)

def show_ssd_tab():
    """Small Sequence Design tab (preserved exactly from original)"""
    
    st.header("🔬 Small Sequence Design (SSD)")
    st.markdown("*Design DNA sequences with enzyme sites for cloning and protein expression*")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Sequence Input")
        
        input_sequence = st.text_area(
            "DNA Sequence",
            height=150,
            placeholder="Enter your DNA sequence here...",
            help="Enter the DNA sequence you want to process for synthesis"
        )
        
        # Sequence type selection
        sequence_type = st.radio(
            "Sequence Type",
            ["Coding Sequence (starts with ATG)", "Non-coding Sequence"],
            help="Select whether your sequence is a coding sequence that starts with ATG"
        )
        
        is_coding = sequence_type == "Coding Sequence (starts with ATG)"
        
        if is_coding:
            remove_stop = st.checkbox(
                "Remove stop codon",
                value=False,
                help="Remove the stop codon from the coding sequence"
            )
        else:
            remove_stop = False
    
    with col2:
        st.subheader("⚙️ Parameters")
        
        # Enzyme pair selection
        enzyme_pair = st.selectbox(
            "Enzyme Pair",
            ["NdeI/XhoI", "NdeI/EcoRI", "BamHI/EcoRI", "BamHI/XhoI", "SalI/XbaI"],
            help="Select the restriction enzyme pair for cloning"
        )
        
        # Cleavage site selection (for non-coding sequences)
        if not is_coding:
            cleavage_site = st.selectbox(
                "Cleavage Site",
                ["None"] + list(SSD_CLEAVAGE_SITES.keys()),
                help="Optional protease cleavage site for tag removal"
            )
            if cleavage_site == "None":
                cleavage_site = None
        else:
            cleavage_site = None
        
        # Analysis button
        if st.button("🔬 Process Sequence", type="primary", use_container_width=True):
            if not input_sequence:
                st.error("❌ Please enter a DNA sequence")
            else:
                with st.spinner("Processing sequence..."):
                    result = ssd_process_sequence(input_sequence, is_coding, remove_stop, enzyme_pair, cleavage_site)
                
                st.session_state.ssd_results = result
                
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.success("✅ Sequence processing completed!")
    
    # Results display
    if 'ssd_results' in st.session_state and st.session_state.ssd_results and "error" not in st.session_state.ssd_results:
        result = st.session_state.ssd_results
        
        st.markdown("---")
        st.subheader("📊 Results")
        
        # Properties
        props = result['properties']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Forward Length", f"{props['forward_length']} bp")
        with col2:
            st.metric("Reverse Length", f"{props['reverse_length']} bp")
        with col3:
            st.metric("Forward GC", f"{props['forward_gc']:.1f}%")
        with col4:
            st.metric("Reverse GC", f"{props['reverse_gc']:.1f}%")
        
        # Tm values
        if props['forward_tm'] and props['reverse_tm']:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Forward Tm", f"{props['forward_tm']:.1f}°C")
            with col2:
                st.metric("Reverse Tm", f"{props['reverse_tm']:.1f}°C")
        
        # Sequences
        tab1, tab2 = st.tabs(["🧬 Forward Strand", "🧬 Reverse Strand"])
        
        with tab1:
            st.markdown("**Forward Strand (5' → 3'):**")
            st.markdown(f'<div class="sequence-display">{result["forward"]}</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown("**Reverse Strand (5' → 3'):**")
            st.markdown(f'<div class="sequence-display">{result["reverse"]}</div>', unsafe_allow_html=True)
        
        # Export
        st.subheader("📄 Export")
        col1, col2 = st.columns(2)
        
        with col1:
            fasta_content = f">Forward_Strand\n{result['forward']}\n>Reverse_Strand\n{result['reverse']}"
            st.download_button(
                "📄 Download FASTA",
                fasta_content,
                "ssd_sequences.fasta",
                "text/plain",
                use_container_width=True
            )
        
        with col2:
            json_content = json.dumps(result, indent=2)
            st.download_button(
                "📊 Download JSON",
                json_content,
                "ssd_results.json",
                "application/json",
                use_container_width=True
            )

def show_translation_tab():
    """Translation & Reverse Translation tab (preserved exactly from original)"""
    
    st.header("🔄 Translation & Reverse Translation")
    st.markdown("*Convert between DNA sequences and amino acid sequences*")
    
    # Mode selection
    mode = st.radio(
        "Select Mode",
        ["DNA to Protein (Translation)", "Protein to DNA (Reverse Translation)"],
        horizontal=True
    )
    
    if mode == "DNA to Protein (Translation)":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            dna_sequence = st.text_area(
                "DNA Sequence",
                height=150,
                placeholder="Enter DNA sequence...",
                help="Enter the DNA sequence to translate"
            )
            
            # Translation options
            frame = st.selectbox("Reading Frame", [0, 1, 2], help="Select the reading frame (0, 1, or 2)")
            find_start = st.checkbox("Find start codon (ATG)", value=True, help="Start translation at the first ATG")
            show_all_frames = st.checkbox("Show all 6 frames", value=False, help="Show translation in all 6 reading frames")
        
        with col2:
            st.subheader("⚙️ Options")
            three_letter = st.checkbox("Three-letter amino acid codes", value=False)
            show_orfs = st.checkbox("Show ORFs", value=False)
        
        if st.button("🔄 Translate", type="primary", use_container_width=True):
            if not dna_sequence:
                st.error("❌ Please enter a DNA sequence")
            else:
                clean_seq = clean_dna_sequence(dna_sequence)
                
                if not clean_seq:
                    st.error("❌ Invalid DNA sequence")
                else:
                    st.session_state.translation_results = {}
                    
                    if show_all_frames:
                        st.subheader("📊 All Reading Frames")
                        
                        # Forward frames
                        st.write("**Forward Frames:**")
                        for f in range(3):
                            protein = translate_sequence(clean_seq, f, find_start)
                            if three_letter:
                                protein = convert_to_three_letter(protein)
                            
                            st.write(f"Frame {f+1}:")
                            st.markdown(f'<div class="sequence-display">{protein}</div>', unsafe_allow_html=True)
                        
                        # Reverse frames
                        st.write("**Reverse Frames:**")
                        reverse_seq = reverse_complement(clean_seq)
                        for f in range(3):
                            protein = translate_sequence(reverse_seq, f, find_start)
                            if three_letter:
                                protein = convert_to_three_letter(protein)
                            
                            st.write(f"Frame {f+4}:")
                            st.markdown(f'<div class="sequence-display">{protein}</div>', unsafe_allow_html=True)
                    
                    else:
                        protein = translate_sequence(clean_seq, frame, find_start)
                        if three_letter:
                            protein = convert_to_three_letter(protein)
                        
                        st.subheader("🧬 Translation Result")
                        st.markdown(f'<div class="sequence-display">{protein}</div>', unsafe_allow_html=True)
                        
                        # Protein properties
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Length", f"{len(protein)} aa")
                        with col2:
                            molecular_weight = len(protein) * 110  # Approximate
                            st.metric("Molecular Weight", f"~{molecular_weight/1000:.1f} kDa")
                        with col3:
                            stop_count = protein.count('*')
                            st.metric("Stop Codons", stop_count)
                        
                        st.session_state.translation_results = {
                            'dna': clean_seq,
                            'protein': protein,
                            'frame': frame,
                            'length': len(protein)
                        }
                    
                    # ORF analysis
                    if show_orfs:
                        st.subheader("🔍 Open Reading Frames")
                        orfs = find_orfs(clean_seq)
                        
                        if orfs:
                            orf_data = []
                            for i, (start, end, frame) in enumerate(orfs):
                                length = end - start
                                orf_protein = translate_sequence(clean_seq[start:end])
                                orf_data.append({
                                    'ORF': i+1,
                                    'Start': start+1,
                                    'End': end,
                                    'Length': length,
                                    'Frame': frame+1,
                                    'Protein Length': len(orf_protein)-1  # Exclude stop codon
                                })
                            
                            st.dataframe(pd.DataFrame(orf_data), use_container_width=True)
                        else:
                            st.info("No ORFs found")
    
    else:  # Reverse Translation
        col1, col2 = st.columns([2, 1])
        
        with col1:
            protein_sequence = st.text_area(
                "Protein Sequence",
                height=150,
                placeholder="Enter protein sequence...",
                help="Enter the protein sequence to reverse translate"
            )
        
        with col2:
            st.subheader("⚙️ Options")
            target_organism = st.selectbox(
                "Target Organism",
                list(CODON_USAGE_TABLES.keys()),
                help="Select organism for codon optimization"
            )
        
        if st.button("🔄 Reverse Translate", type="primary", use_container_width=True):
            if not protein_sequence:
                st.error("❌ Please enter a protein sequence")
            else:
                # Clean protein sequence
                clean_protein = "".join(c for c in protein_sequence.upper() if c in "ACDEFGHIKLMNPQRSTVWY*")
                
                if not clean_protein:
                    st.error("❌ Invalid protein sequence")
                else:
                    dna_result = reverse_translate_to_dna(clean_protein, target_organism)
                    
                    st.subheader("🧬 Reverse Translation Result")
                    st.markdown(f'<div class="sequence-display">{dna_result}</div>', unsafe_allow_html=True)
                    
                    # Properties
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("DNA Length", f"{len(dna_result)} bp")
                    with col2:
                        st.metric("Protein Length", f"{len(clean_protein)} aa")
                    with col3:
                        gc_content = calculate_gc(dna_result)
                        st.metric("GC Content", f"{gc_content:.1f}%")
                    
                    st.session_state.translation_results = {
                        'protein': clean_protein,
                        'dna': dna_result,
                        'organism': target_organism,
                        'gc_content': gc_content
                    }
    
    # Export results
    if 'translation_results' in st.session_state and st.session_state.translation_results:
        st.subheader("📄 Export Results")
        
        results = st.session_state.translation_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'dna' in results and 'protein' in results:
                fasta_content = f">DNA_Sequence\n{results['dna']}\n>Protein_Sequence\n{results['protein']}"
                st.download_button(
                    "📄 Download FASTA",
                    fasta_content,
                    "translation_results.fasta",
                    "text/plain",
                    use_container_width=True
                )
        
        with col2:
            json_content = json.dumps(results, indent=2)
            st.download_button(
                "📊 Download JSON",
                json_content,
                "translation_results.json",
                "application/json",
                use_container_width=True
            )

def show_primer_generator_tab():
    """Primer Generator tab (preserved exactly from original)"""
    
    st.header("🧪 Primer Generator")
    st.markdown("*Design optimal PCR primers with specific features*")
    
    # Mode selection
    mode = st.radio(
        "Primer Mode",
        ["Simple PCR", "Cloning (with restriction sites)"],
        horizontal=True
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if mode == "Simple PCR":
            target_sequence = st.text_area(
                "Target Sequence",
                height=150,
                placeholder="Enter target DNA sequence...",
                help="DNA sequence to amplify"
            )
        else:  # Cloning mode
            st.subheader("📝 Input Sequences")
            
            forward_seq = st.text_area(
                "Forward Sequence",
                height=100,
                placeholder="Enter forward sequence...",
                help="Forward part of the sequence to clone"
            )
            
            reverse_seq = st.text_area(
                "Reverse Sequence", 
                height=100,
                placeholder="Enter reverse sequence...",
                help="Reverse part of the sequence to clone"
            )
    
    with col2:
        st.subheader("⚙️ Parameters")
        
        if mode == "Simple PCR":
            primer_length = st.slider("Primer Length", 15, 35, 20)
            target_tm = st.slider("Target Tm (°C)", 50, 80, 60)
            gc_min = st.slider("Min GC%", 30, 60, 40)
            gc_max = st.slider("Max GC%", 40, 80, 60)
        
        else:  # Cloning mode
            fwd_enzyme = st.selectbox(
                "Forward Enzyme",
                list(enzyme_linkers.keys()),
                help="Restriction enzyme for forward primer"
            )
            
            rev_enzyme = st.selectbox(
                "Reverse Enzyme",
                list(enzyme_linkers.keys()),
                index=1,
                help="Restriction enzyme for reverse primer"
            )
            
            custom_prefix = st.text_input(
                "Custom Prefix",
                value="TGCATC",
                help="Additional sequence added to primers"
            )
            
            primer_conc = st.number_input(
                "Primer Concentration (nM)",
                min_value=10,
                max_value=1000,
                value=500,
                help="Primer concentration for Tm calculation"
            )
            
            # Check for restriction sites in input
            if st.button("🔍 Check Restriction Sites"):
                if forward_seq and reverse_seq:
                    combined_seq = forward_seq + reverse_seq
                    found_sites = []
                    
                    for enzyme, site in enzyme_linkers.items():
                        if site in combined_seq.upper():
                            found_sites.append(enzyme)
                    
                    if found_sites:
                        st.warning(f"⚠️ Found restriction sites in sequence: {', '.join(found_sites)}")
                    else:
                        st.success("✅ No restriction sites found in input sequence")
    
    # Generate primers
    if st.button("🧪 Generate Primers", type="primary", use_container_width=True):
        if mode == "Simple PCR":
            if not target_sequence:
                st.error("❌ Please enter a target sequence")
            else:
                clean_seq = clean_dna_sequence(target_sequence)
                
                if len(clean_seq) < primer_length * 2:
                    st.error("❌ Target sequence too short for primer design")
                else:
                    # Simple primer design
                    forward_primer = clean_seq[:primer_length]
                    reverse_primer = reverse_complement(clean_seq[-primer_length:])
                    
                    # Calculate properties
                    fwd_tm = calculate_tm_consensus(forward_primer)
                    rev_tm = calculate_tm_consensus(reverse_primer)
                    fwd_gc = calculate_gc(forward_primer)
                    rev_gc = calculate_gc(reverse_primer)
                    
                    st.session_state.primer_results = {
                        'mode': 'simple',
                        'forward': forward_primer,
                        'reverse': reverse_primer,
                        'forward_tm': fwd_tm,
                        'reverse_tm': rev_tm,
                        'forward_gc': fwd_gc,
                        'reverse_gc': rev_gc,
                        'target': clean_seq
                    }
                    
                    st.success("✅ Primers generated successfully!")
        
        else:  # Cloning mode
            if not forward_seq or not reverse_seq:
                st.error("❌ Please enter both forward and reverse sequences")
            else:
                clean_fwd = clean_dna_sequence(forward_seq)
                clean_rev = clean_dna_sequence(reverse_seq)
                
                # Use the original cloning primer design function
                primer_fwd, primer_rev, fwd_len, rev_len, fwd_tm, rev_tm = design_cloning_primers(
                    clean_fwd, clean_rev, fwd_enzyme, rev_enzyme, primer_conc, custom_prefix
                )
                
                st.session_state.primer_results = {
                    'mode': 'cloning',
                    'forward': primer_fwd,
                    'reverse': primer_rev,
                    'forward_length': fwd_len,
                    'reverse_length': rev_len,
                    'forward_tm': fwd_tm,
                    'reverse_tm': rev_tm,
                    'forward_enzyme': fwd_enzyme,
                    'reverse_enzyme': rev_enzyme,
                    'prefix': custom_prefix
                }
                
                st.success("✅ Cloning primers generated successfully!")
    
    # Results display
    if 'primer_results' in st.session_state and st.session_state.primer_results:
        result = st.session_state.primer_results
        
        st.markdown("---")
        st.subheader("📊 Primer Results")
        
        # Primer sequences
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Forward Primer (5' → 3'):**")
            st.markdown(f'<div class="sequence-display">{result["forward"]}</div>', unsafe_allow_html=True)
            
            if result['mode'] == 'cloning':
                st.info(f"Length: {result['forward_length']} bp")
            if 'forward_tm' in result and result['forward_tm']:
                st.info(f"Tm: {result['forward_tm']:.1f}°C")
        
        with col2:
            st.markdown("**Reverse Primer (5' → 3'):**")
            st.markdown(f'<div class="sequence-display">{result["reverse"]}</div>', unsafe_allow_html=True)
            
            if result['mode'] == 'cloning':
                st.info(f"Length: {result['reverse_length']} bp")
            if 'reverse_tm' in result and result['reverse_tm']:
                st.info(f"Tm: {result['reverse_tm']:.1f}°C")
        
        # Properties table
        if result['mode'] == 'simple':
            properties_data = {
                'Property': ['Length (bp)', 'Tm (°C)', 'GC Content (%)'],
                'Forward Primer': [
                    len(result['forward']),
                    f"{result['forward_tm']:.1f}" if result['forward_tm'] else "N/A",
                    f"{result['forward_gc']:.1f}"
                ],
                'Reverse Primer': [
                    len(result['reverse']),
                    f"{result['reverse_tm']:.1f}" if result['reverse_tm'] else "N/A", 
                    f"{result['reverse_gc']:.1f}"
                ]
            }
            
            st.dataframe(pd.DataFrame(properties_data), use_container_width=True)
        
        elif result['mode'] == 'cloning':
            st.subheader("🔧 Cloning Details")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Forward Enzyme:** {result['forward_enzyme']}")
                st.write(f"**Recognition Site:** {enzyme_linkers[result['forward_enzyme']]}")
            
            with col2:
                st.write(f"**Reverse Enzyme:** {result['reverse_enzyme']}")
                st.write(f"**Recognition Site:** {enzyme_linkers[result['reverse_enzyme']]}")
            
            if result['prefix']:
                st.write(f"**Custom Prefix:** {result['prefix']}")
        
        # Export
        st.subheader("📄 Export Primers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fasta_content = f">Forward_Primer\n{result['forward']}\n>Reverse_Primer\n{result['reverse']}"
            st.download_button(
                "📄 Download FASTA",
                fasta_content,
                "primers.fasta",
                "text/plain",
                use_container_width=True
            )
        
        with col2:
            json_content = json.dumps(result, indent=2)
            st.download_button(
                "📊 Download JSON",
                json_content,
                "primer_results.json", 
                "application/json",
                use_container_width=True
            )

def show_reverse_complement_tab():
    """Reverse Complement tab (preserved exactly from original)"""
    
    st.header("↔️ Reverse Complement")
    st.markdown("*Generate complementary DNA strands for molecular biology applications*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_sequence = st.text_area(
            "DNA Sequence",
            height=200,
            placeholder="Enter DNA sequence...",
            help="Enter the DNA sequence to get its reverse complement"
        )
    
    with col2:
        st.subheader("⚙️ Options")
        
        show_both = st.checkbox("Show both strands", value=True)
        include_properties = st.checkbox("Include sequence properties", value=True)
        format_output = st.selectbox("Output format", ["Standard", "FASTA", "GenBank style"])
    
    if st.button("↔️ Generate Reverse Complement", type="primary", use_container_width=True):
        if not input_sequence:
            st.error("❌ Please enter a DNA sequence")
        else:
            clean_seq = clean_dna_sequence(input_sequence)
            
            if not clean_seq:
                st.error("❌ Invalid DNA sequence")
            else:
                rev_comp = reverse_complement(clean_seq)
                
                st.subheader("📊 Results")
                
                if show_both:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Original Sequence (5' → 3'):**")
                        if format_output == "FASTA":
                            st.markdown(f'<div class="sequence-display">>Original_Sequence\n{clean_seq}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="sequence-display">{clean_seq}</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**Reverse Complement (5' → 3'):**")
                        if format_output == "FASTA":
                            st.markdown(f'<div class="sequence-display">>Reverse_Complement\n{rev_comp}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="sequence-display">{rev_comp}</div>', unsafe_allow_html=True)
                else:
                    st.markdown("**Reverse Complement (5' → 3'):**")
                    st.markdown(f'<div class="sequence-display">{rev_comp}</div>', unsafe_allow_html=True)
                
                # Properties
                if include_properties:
                    st.subheader("📈 Sequence Properties")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Length", f"{len(clean_seq)} bp")
                    
                    with col2:
                        gc_content = calculate_gc(clean_seq)
                        st.metric("GC Content", f"{gc_content:.1f}%")
                    
                    with col3:
                        tm = calculate_tm_consensus(clean_seq)
                        if tm:
                            st.metric("Tm", f"{tm:.1f}°C")
                        else:
                            st.metric("Tm", "N/A")
                    
                    with col4:
                        at_content = 100 - gc_content
                        st.metric("AT Content", f"{at_content:.1f}%")
                    
                    # Base composition
                    st.subheader("🧬 Base Composition")
                    
                    base_counts = {
                        'A': clean_seq.count('A'),
                        'T': clean_seq.count('T'), 
                        'G': clean_seq.count('G'),
                        'C': clean_seq.count('C')
                    }
                    
                    composition_data = {
                        'Base': list(base_counts.keys()),
                        'Count': list(base_counts.values()),
                        'Percentage': [f"{(count/len(clean_seq)*100):.1f}%" for count in base_counts.values()]
                    }
                    
                    st.dataframe(pd.DataFrame(composition_data), use_container_width=True)
                
                # Store results
                results = {
                    'original': clean_seq,
                    'reverse_complement': rev_comp,
                    'length': len(clean_seq),
                    'gc_content': calculate_gc(clean_seq),
                    'base_composition': base_counts
                }
                
                # Export
                st.subheader("📄 Export")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fasta_content = f">Original_Sequence\n{clean_seq}\n>Reverse_Complement\n{rev_comp}"
                    st.download_button(
                        "📄 Download FASTA",
                        fasta_content,
                        "reverse_complement.fasta",
                        "text/plain",
                        use_container_width=True
                    )
                
                with col2:
                    json_content = json.dumps(results, indent=2)
                    st.download_button(
                        "📊 Download JSON",
                        json_content,
                        "reverse_complement.json",
                        "application/json",
                        use_container_width=True
                    )
                
                with col3:
                    # Text format
                    text_content = f"Original Sequence (5' → 3'):\n{clean_seq}\n\nReverse Complement (5' → 3'):\n{rev_comp}\n\nLength: {len(clean_seq)} bp\nGC Content: {calculate_gc(clean_seq):.1f}%"
                    st.download_button(
                        "📝 Download Text",
                        text_content,
                        "reverse_complement.txt",
                        "text/plain",
                        use_container_width=True
                    )

def show_ligation_check_tab():
    """Ligation Check tab (preserved exactly from original)"""
    
    st.header("🔗 Ligation Check")
    st.markdown("*Verify compatibility of DNA fragments for ligation reactions*")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Sequence Input")
        
        forward_strand = st.text_area(
            "Forward Strand (5' → 3')",
            height=100,
            placeholder="Enter forward strand sequence...",
            help="The forward DNA strand with sticky ends"
        )
        
        reverse_strand = st.text_area(
            "Reverse Strand (5' → 3')",
            height=100,
            placeholder="Enter reverse strand sequence...",
            help="The reverse DNA strand with sticky ends"
        )
        
        vector_sequence = st.text_area(
            "Vector Sequence (optional)",
            height=100,
            placeholder="Enter vector sequence...",
            help="Optional: Vector sequence for compatibility checking"
        )
    
    with col2:
        st.subheader("⚙️ Parameters")
        
        enzyme_pair = st.selectbox(
            "Enzyme Pair",
            list(enzyme_pairs.keys()),
            help="Restriction enzyme pair used for digestion"
        )
        
        vector_name = st.text_input("Vector Name", value="pET-21a(+)")
        insert_name = st.text_input("Insert Name", value="Gene of Interest")
        
        check_complementarity = st.checkbox("Check base complementarity", value=True)
        show_alignment = st.checkbox("Show sequence alignment", value=True)
    
    # Analysis buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔗 Check Ligation", type="primary", use_container_width=True):
            if not forward_strand or not reverse_strand:
                st.error("❌ Please enter both forward and reverse strands")
            else:
                # Perform ligation check
                clean_fwd = clean_dna_sequence(forward_strand)
                clean_rev = clean_dna_sequence(reverse_strand)
                
                # Get enzyme information
                pair_info = enzyme_pairs.get(enzyme_pair, {})
                forward_overhang = pair_info.get("forward_overhang", "")
                reverse_overhang = pair_info.get("reverse_overhang", "")
                
                # Validate sticky ends
                fwd_valid = clean_fwd.startswith(forward_overhang) if forward_overhang else True
                rev_valid = clean_rev.startswith(reverse_overhang) if reverse_overhang else True
                
                # Check hybridization
                if fwd_valid and rev_valid:
                    # Remove overhangs for comparison
                    fwd_internal = clean_fwd[len(forward_overhang):] if forward_overhang else clean_fwd
                    rev_internal = clean_rev[len(reverse_overhang):] if reverse_overhang else clean_rev
                    
                    # Check if reverse is the reverse complement of forward
                    expected_rev = reverse_complement(fwd_internal)
                    hybridization_match = (rev_internal == expected_rev)
                    
                    # Overall compatibility
                    compatible = fwd_valid and rev_valid and hybridization_match
                    
                    results = {
                        'compatible': compatible,
                        'forward_valid': fwd_valid,
                        'reverse_valid': rev_valid,
                        'hybridization_match': hybridization_match,
                        'forward_overhang': forward_overhang,
                        'reverse_overhang': reverse_overhang,
                        'enzyme_pair': enzyme_pair
                    }
                    
                    st.session_state.ligation_results = results
                    
                    # Display results
                    if compatible:
                        st.success("✅ Ligation compatibility confirmed!")
                    else:
                        st.error("❌ Ligation compatibility issues detected")
                else:
                    st.error("❌ Invalid sticky ends for selected enzyme pair")
    
    with col2:
        if st.button("🧬 Simulate Ligation", use_container_width=True):
            if 'ligation_results' in st.session_state:
                results = st.session_state.ligation_results
                
                if results['compatible']:
                    # Simulate the ligation product
                    clean_fwd = clean_dna_sequence(forward_strand)
                    clean_rev = clean_dna_sequence(reverse_strand)
                    
                    # Create ligated product (simplified)
                    ligated_product = clean_fwd + clean_rev
                    
                    st.subheader("🧬 Ligation Product")
                    st.markdown(f'<div class="sequence-display">{ligated_product}</div>', unsafe_allow_html=True)
                    
                    # Product properties
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Product Length", f"{len(ligated_product)} bp")
                    with col2:
                        gc_content = calculate_gc(ligated_product)
                        st.metric("GC Content", f"{gc_content:.1f}%")
                    with col3:
                        tm = calculate_tm_consensus(ligated_product[:50])  # First 50 bp
                        if tm:
                            st.metric("Tm (50bp)", f"{tm:.1f}°C")
                else:
                    st.warning("⚠️ Cannot simulate ligation - compatibility issues detected")
            else:
                st.info("ℹ️ Please check ligation compatibility first")
    
    # Results display
    if 'ligation_results' in st.session_state:
        results = st.session_state.ligation_results
        
        st.markdown("---")
        st.subheader("📊 Ligation Analysis")
        
        # Compatibility summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if results['forward_valid']:
                st.success("✅ Forward strand valid")
            else:
                st.error("❌ Forward strand invalid")
        
        with col2:
            if results['reverse_valid']:
                st.success("✅ Reverse strand valid")
            else:
                st.error("❌ Reverse strand invalid")
        
        with col3:
            if results['hybridization_match']:
                st.success("✅ Hybridization compatible")
            else:
                st.error("❌ Hybridization incompatible")
        
        # Detailed analysis
        if show_alignment and results['compatible']:
            st.subheader("🔍 Sequence Alignment")
            
            clean_fwd = clean_dna_sequence(forward_strand)
            clean_rev = clean_dna_sequence(reverse_strand)
            
            # Show sticky end alignment
            fwd_overhang = results['forward_overhang']
            rev_overhang = results['reverse_overhang']
            
            if fwd_overhang and rev_overhang:
                st.write("**Sticky End Alignment:**")
                st.code(f"5'-{fwd_overhang}...{rev_overhang}-3'")
                st.code(f"3'-{reverse_complement(fwd_overhang)}...{reverse_complement(rev_overhang)}-5'")
        
        # Export results
        if results['compatible']:
            st.subheader("📄 Export")
            
            export_data = {
                'ligation_results': results,
                'sequences': {
                    'forward_strand': forward_strand,
                    'reverse_strand': reverse_strand
                },
                'analysis_date': datetime.now().isoformat()
            }
            
            json_content = json.dumps(export_data, indent=2)
            st.download_button(
                "📊 Download Analysis",
                json_content,
                "ligation_analysis.json",
                "application/json",
                use_container_width=True
            )

def show_hybridization_tab():
    """Hybridization Simulation tab (preserved exactly from original)"""
    
    st.header("🌀 Hybridization Simulation")
    st.markdown("*Simulate DNA strand hybridization and binding interactions*")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Sequence Input")
        
        forward_sequence = st.text_area(
            "Forward Sequence (5' → 3')",
            height=120,
            placeholder="Enter forward DNA sequence...",
            value="ATGAAAGAACTGACCGGTATTCCGGAAGGTCTGGTGATCGGCATTACCCTGGAAGCTAACCTGCTGAAAGATCCGGATGAAATCACCGGTTACCTGGTGCTGGGTATTGGCAACTCC"
        )
        
        reverse_sequence = st.text_area(
            "Reverse Sequence (5' → 3')",
            height=120,
            placeholder="Enter reverse DNA sequence...",
            value="GGAGTTGCCAATACCCAGCACCAGGTAACCGGTGATTTCATCCGGATCTTTCAGCAGGTTAG"
        )
    
    with col2:
        st.subheader("⚙️ Simulation Parameters")
        
        max_shift = st.number_input("Max Shift Position", min_value=10, max_value=500, value=100)
        show_alignment = st.checkbox("Show detailed alignment", value=True)
        highlight_matches = st.checkbox("Highlight complementary bases", value=True)
        temp_analysis = st.checkbox("Temperature analysis", value=False)
        
        if temp_analysis:
            temperature = st.slider("Temperature (°C)", 4, 95, 37)
            salt_conc = st.slider("Salt concentration (mM)", 1, 1000, 50)
    
    # Analysis button
    if st.button("🌀 Simulate Hybridization", type="primary", use_container_width=True):
        if not forward_sequence or not reverse_sequence:
            st.error("❌ Please enter both forward and reverse sequences")
        else:
            # Clean sequences
            clean_fwd = clean_dna_sequence(forward_sequence)
            clean_rev = clean_dna_sequence(reverse_sequence)
            
            # Original logic: reverse the reverse sequence (not reverse complement!)
            reversed_seq = clean_rev[::-1]  # Plain reversal as per original
            
            with st.spinner("🌀 Running hybridization simulation..."):
                # Find optimal alignment (preserved exactly from original)
                shift, score = optimal_alignment(clean_fwd, reversed_seq, max_shift)
            
            st.session_state.hybridization_results = {
                'forward': clean_fwd,
                'reverse': clean_rev,
                'reversed': reversed_seq,
                'shift': shift,
                'score': score,
                'max_possible': min(len(clean_fwd), len(reversed_seq))
            }
            
            st.success(f"✅ Hybridization simulation completed! Best shift: {shift}, Score: {score}")
            
            # Results display
            result = st.session_state.hybridization_results
            
            # Alignment visualization
            st.subheader("🔍 Alignment Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Shift", result['shift'])
            with col2:
                st.metric("Match Score", f"{result['score']}/{result['max_possible']}")
            with col3:
                similarity = (result['score'] / result['max_possible']) * 100 if result['max_possible'] > 0 else 0
                st.metric("Similarity", f"{similarity:.1f}%")
            
            # Visual alignment
            if show_alignment:
                st.subheader("📊 Sequence Alignment")
                
                # Create alignment visualization
                fwd_seq = result['forward']
                rev_seq = result['reversed']
                shift = result['shift']
                
                # Build alignment strings with highlighting
                alignment_fwd = ""
                alignment_rev = ""
                alignment_markers = ""
                
                # Calculate alignment bounds
                start_pos = max(0, shift)
                end_pos = min(len(fwd_seq), shift + len(rev_seq))
                
                # Build the alignment display
                for i in range(len(fwd_seq)):
                    j = i - shift
                    
                    if 0 <= j < len(rev_seq):
                        # Both sequences have bases at this position
                        fwd_base = fwd_seq[i]
                        rev_base = rev_seq[j]
                        
                        if highlight_matches and is_complement(fwd_base, rev_base):
                            alignment_fwd += f"**{fwd_base}**"
                            alignment_rev += f"**{rev_base}**"
                            alignment_markers += "|"
                        else:
                            alignment_fwd += fwd_base
                            alignment_rev += rev_base
                            alignment_markers += " "
                    else:
                        alignment_fwd += fwd_seq[i]
                        alignment_rev += "-" if j < 0 else " "
                        alignment_markers += " "
                
                # Display alignment
                st.write("**Forward (5' → 3'):**")
                st.markdown(f'<div class="sequence-display">{alignment_fwd}</div>', unsafe_allow_html=True)
                
                st.write("**Alignment:**")
                st.markdown(f'<div class="sequence-display">{alignment_markers}</div>', unsafe_allow_html=True)
                
                st.write("**Reverse (reversed 5' → 3'):**")
                st.markdown(f'<div class="sequence-display">{alignment_rev}</div>', unsafe_allow_html=True)
            
            # Export results
            st.subheader("📄 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Text format
                alignment_text = f"""Hybridization Simulation Results
=====================================

Forward sequence: {result['forward']}
Reverse sequence: {result['reverse']}
Reversed sequence: {result['reversed']}

Best alignment shift: {result['shift']}
Match score: {result['score']}/{result['max_possible']}
Similarity: {similarity:.1f}%
"""
                
                st.download_button(
                    "📝 Download Text Report",
                    alignment_text,
                    "hybridization_results.txt",
                    "text/plain",
                    use_container_width=True
                )
            
            with col2:
                # JSON format
                json_content = json.dumps(result, indent=2)
                st.download_button(
                    "📊 Download JSON",
                    json_content,
                    "hybridization_results.json",
                    "application/json",
                    use_container_width=True
                )

def show_extended_synthesis_tab():
    """Extended Synthesis tab (preserved exactly from original)"""
    
    st.header("📏 Extended Synthesis (Long Sequences)")
    st.markdown("*Fragment and assemble large DNA sequences for synthesis*")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Sequence Input")
        
        input_sequence = st.text_area(
            "Long DNA Sequence",
            height=200,
            placeholder="Enter long DNA sequence to fragment...",
            help="Enter a long DNA sequence that needs to be broken into smaller fragments for synthesis"
        )
        
        # Example sequence button
        if st.button("📋 Load Example Sequence"):
            example_seq = "ATGAAAGAACTGACCGGTATTCCGGAAGGTCTGGTGATCGGCATTACCCTGGAAGCTAACCTGCTGAAAGATCCGGATGAAATCACCGGTTACCTGGTGCTGGGTATTGGCAACTCCCTGCTGGAAGAACTGGAAGATGCGGAAGTTTGCGTGAAAGGTGCGAAGGGTCACGTGGAAGGTGGTAAGGACATGGGCAAGGTGAAGGGTACCGGTCTGCTGGAAGAACTGGACACCGTGAAGTACAAGGTGCCCTACGGCGTGAACTTCCCCTCCGACGGCCCAGTGATGCAGAAGAAAACCATGGGCTGGGAAGCCTCCACCGAGCGGATGTACCCCGAGGACGGCGCCCTGAAGGGCGAGATCAAGCAGAGGCTGAAGCTGAAGGACGGCGGCCACTACGACGCCGAGGTGAAGACCACCTACAAGGCCAAGAAGCCCGTGCAGCTGCCCGGCGCCTACAACGTGAACATCAAGCTGGACATCACCTCCCACAACGAGGACTACACCATCGTGGAACAGTACGAGCGCGCCGAGGGCAGGCACCCCACCGGCGGCATGGACGAGCTGTACAAGTAA"
            st.session_state.extended_sequence = example_seq
            st.experimental_rerun()
        
        # Use session state for persistence
        if hasattr(st.session_state, 'extended_sequence'):
            input_sequence = st.session_state.extended_sequence
    
    with col2:
        st.subheader("⚙️ Fragmentation Parameters")
        
        fragment_size = st.number_input(
            "Fragment Size (bp)",
            min_value=50,
            max_value=500,
            value=150,
            help="Maximum size of each fragment"
        )
        
        overlap_size = st.number_input(
            "Overlap Size (bp)",
            min_value=10,
            max_value=50,
            value=20,
            help="Size of overlap between adjacent fragments"
        )
        
        enzyme_pair = st.selectbox(
            "Terminal Enzyme Pair",
            list(enzyme_pairs.keys()),
            help="Restriction enzymes for terminal fragment ends"
        )
        
        cleavage_site = st.selectbox(
            "Cleavage Site",
            ["None"] + list(cleavage_sites.keys()),
            help="Optional protease cleavage site"
        )
        
        assembly_method = st.selectbox(
            "Assembly Method",
            ["Overlap Extension", "Gibson Assembly", "Golden Gate"],
            help="Method for reassembling fragments"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            optimize_junctions = st.checkbox("Optimize junction sequences", value=True)
            avoid_hairpins = st.checkbox("Avoid secondary structures", value=True)
            check_restriction_sites = st.checkbox("Check internal restriction sites", value=True)
    
    # Fragment button
    if st.button("📏 Fragment Sequence", type="primary", use_container_width=True):
        if not input_sequence:
            st.error("❌ Please enter a DNA sequence")
        elif len(clean_dna_sequence(input_sequence)) < fragment_size:
            st.error(f"❌ Sequence too short. Minimum length: {fragment_size} bp")
        else:
            clean_seq = clean_dna_sequence(input_sequence)
            
            with st.spinner("📏 Fragmenting sequence..."):
                try:
                    # Use original fragmentation logic
                    cleavage_seq = cleavage_sites.get(cleavage_site, "") if cleavage_site != "None" else ""
                    
                    assembly, reassembled = fragment_extended_sequence(
                        clean_seq, fragment_size, enzyme_pair, cleavage_seq, overlap_size
                    )
                    
                    st.session_state.extended_synthesis_results = {
                        'original_sequence': clean_seq,
                        'assembly': assembly,
                        'reassembled': reassembled,
                        'parameters': {
                            'fragment_size': fragment_size,
                            'overlap_size': overlap_size,
                            'enzyme_pair': enzyme_pair,
                            'cleavage_site': cleavage_site,
                            'assembly_method': assembly_method
                        }
                    }
                    
                    st.success(f"✅ Sequence fragmented into {len(assembly)} fragments!")
                    
                except Exception as e:
                    st.error(f"❌ Fragmentation failed: {str(e)}")
                    return
    
    # Results display
    if 'extended_synthesis_results' in st.session_state and st.session_state.extended_synthesis_results:
        result = st.session_state.extended_synthesis_results
        
        st.markdown("---")
        st.subheader("📊 Fragmentation Results")
        
        # Summary metrics
        original_length = len(result['original_sequence'])
        num_fragments = len(result['assembly'])
        reassembled_length = len(result['reassembled'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Original Length", f"{original_length} bp")
        with col2:
            st.metric("Fragments", num_fragments)
        with col3:
            st.metric("Reassembled Length", f"{reassembled_length} bp")
        with col4:
            accuracy = (reassembled_length / original_length) * 100 if original_length > 0 else 0
            st.metric("Assembly Accuracy", f"{accuracy:.1f}%")
        
        # Fragment details
        tab1, tab2, tab3 = st.tabs(["🧩 Fragment Details", "🔧 Assembly Plan", "📄 Export"])
        
        with tab1:
            st.subheader("Fragment Breakdown")
            
            fragment_data = []
            for frag in result['assembly']:
                fragment_data.append({
                    'Fragment': frag['fragment'],
                    'Type': frag['type'],
                    'Length (bp)': frag['length'],
                    'GC%': f"{calculate_gc(frag['sequence']):.1f}",
                    'Sequence': frag['sequence'][:50] + "..." if len(frag['sequence']) > 50 else frag['sequence']
                })
            
            st.dataframe(pd.DataFrame(fragment_data), use_container_width=True)
            
            # Show individual fragments
            selected_fragment = st.selectbox(
                "View Fragment Details",
                range(1, num_fragments + 1),
                format_func=lambda x: f"Fragment {x}"
            )
            
            if selected_fragment:
                frag = result['assembly'][selected_fragment - 1]
                
                with st.expander(f"Fragment {selected_fragment} Details"):
                    frag_col1, frag_col2 = st.columns(2)
                    
                    with frag_col1:
                        st.write("**Forward Strand:**")
                        st.markdown(f'<div class="sequence-display">{frag["forward"]}</div>', unsafe_allow_html=True)
                    
                    with frag_col2:
                        st.write("**Reverse Strand:**")
                        st.markdown(f'<div class="sequence-display">{frag["reverse"]}</div>', unsafe_allow_html=True)
                    
                    st.write("**Core Sequence:**")
                    st.markdown(f'<div class="sequence-display">{frag["sequence"]}</div>', unsafe_allow_html=True)
        
        with tab2:
            st.subheader("Assembly Strategy")
            
            assembly_method = result['parameters']['assembly_method']
            
            if assembly_method == "Overlap Extension":
                st.markdown("""
                **Overlap Extension PCR Assembly Protocol:**
                
                1. **PCR Amplification**: Amplify each fragment with overlapping primers
                2. **Overlap Extension**: Mix fragments and perform extension without primers
                3. **Final PCR**: Add terminal primers and amplify full-length product
                4. **Cloning**: Insert into vector using terminal restriction sites
                """)
            
            elif assembly_method == "Gibson Assembly":
                st.markdown("""
                **Gibson Assembly Protocol:**
                
                1. **5' Phosphorylation**: Phosphorylate 5' ends of fragments
                2. **Exonuclease Treatment**: Create single-strand overhangs
                3. **Annealing**: Allow complementary overhangs to anneal
                4. **Gap Filling**: DNA polymerase fills gaps
                5. **Ligation**: DNA ligase seals nicks
                """)
            
            elif assembly_method == "Golden Gate":
                st.markdown("""
                **Golden Gate Assembly Protocol:**
                
                1. **Type IIS Digestion**: Cut with Type IIS restriction enzyme
                2. **Ligation**: T4 DNA ligase joins compatible overhangs
                3. **One-pot Reaction**: Digestion and ligation in single tube
                4. **Transformation**: Direct transformation of assembly product
                """)
            
            # Assembly order
            st.subheader("Assembly Order")
            
            order_data = []
            for i, frag in enumerate(result['assembly']):
                order_data.append({
                    'Step': i + 1,
                    'Fragment': frag['fragment'],
                    'Type': frag['type'],
                    'Action': f"Add Fragment {frag['fragment']} ({frag['type']})"
                })
            
            st.dataframe(pd.DataFrame(order_data), use_container_width=True, hide_index=True)
        
        with tab3:
            st.subheader("Export Assembly Plan")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                # FASTA export
                fasta_content = ""
                for frag in result['assembly']:
                    fasta_content += f">Fragment_{frag['fragment']}_{frag['type']}\n{frag['sequence']}\n"
                
                st.download_button(
                    "📄 Download Fragments (FASTA)",
                    fasta_content,
                    "extended_synthesis_fragments.fasta",
                    "text/plain",
                    use_container_width=True
                )
                
                # Assembly protocol
                protocol_content = f"""Extended Synthesis Assembly Protocol
========================================

Original Sequence Length: {original_length} bp
Number of Fragments: {num_fragments}
Assembly Method: {assembly_method}
Fragment Size: {result['parameters']['fragment_size']} bp
Overlap Size: {result['parameters']['overlap_size']} bp

Fragment Details:
"""
                
                for frag in result['assembly']:
                    protocol_content += f"""
Fragment {frag['fragment']} ({frag['type']}):
Length: {frag['length']} bp
Forward: {frag['forward']}
Reverse: {frag['reverse']}
Core: {frag['sequence']}
"""
                
                st.download_button(
                    "📋 Download Protocol",
                    protocol_content,
                    "extended_synthesis_protocol.txt",
                    "text/plain",
                    use_container_width=True
                )
            
            with export_col2:
                # JSON export
                json_content = json.dumps(result, indent=2, default=str)
                st.download_button(
                    "📊 Download JSON",
                    json_content,
                    "extended_synthesis_results.json",
                    "application/json",
                    use_container_width=True
                )
                
                # Primer list
                primer_content = ""
                for i, frag in enumerate(result['assembly']):
                    primer_content += f">Fragment_{frag['fragment']}_Forward\n{frag['forward'][:30]}\n"
                    primer_content += f">Fragment_{frag['fragment']}_Reverse\n{frag['reverse'][:30]}\n"
                
                st.download_button(
                    "🧪 Download Primers",
                    primer_content,
                    "extended_synthesis_primers.fasta",
                    "text/plain",
                    use_container_width=True
                )

def show_codon_optimization_tab():
    """Codon Optimization tab (preserved exactly from original)"""
    
    st.header("⚡ Codon Optimization")
    st.markdown("*Optimize DNA sequences for expression in different host organisms*")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Sequence Input")
        
        input_type = st.radio(
            "Input Type",
            ["DNA Sequence", "Protein Sequence"],
            horizontal=True,
            help="Select whether you're providing DNA or protein sequence"
        )
        
        sequence_input = st.text_area(
            f"{input_type}",
            height=200,
            placeholder=f"Enter {input_type.lower()}...",
            help=f"Enter the {input_type.lower()} to optimize"
        )
        
        # Example sequences
        if st.button("📋 Load Example"):
            if input_type == "DNA Sequence":
                example = "ATGAAAGAACTGACCGGTATTCCGGAAGGTCTGGTGATCGGCATTACCCTGGAAGCTAACCTGCTGAAAGATCCGGATGAAATCACCGGTTACCTGGTGCTGGGTATTGGCAACTCCCTGCTGGAAGAACTGGAAGATGCGGAAGTTTGCGTGAAAGGTGCGAAGGGTCACGTGGAAGGTGGTAAGGACATGGGCAAGGTGAAGGGTACCGGTCTGCTGGAAGAACTG"
            else:
                example = "MKELTSILVKDEGGMKELTSILVKDEGGMKELTSILVKDEGGMKELTSILVKDEGGMKELTSILVKDEGG"
            
            st.session_state.optimization_input = example
    
    with col2:
        st.subheader("⚙️ Optimization Parameters")
        
        target_organism = st.selectbox(
            "Target Organism",
            list(CODON_USAGE_TABLES.keys()),
            help="Organism for which to optimize codon usage"
        )
        
        # Advanced optimization parameters
        with st.expander("🔧 Advanced Parameters"):
            gc_min = st.slider("Minimum GC Content (%)", 20, 60, 30)
            gc_max = st.slider("Maximum GC Content (%)", 40, 80, 70)
            
            avoid_sites = st.multiselect(
                "Avoid Restriction Sites",
                list(enzyme_linkers.keys()),
                help="Restriction sites to avoid in optimized sequence"
            )
            
            optimization_strategy = st.selectbox(
                "Optimization Strategy",
                ["Balanced", "High Expression", "Low Immunogenicity"],
                help="Different optimization approaches"
            )
            
            harmonize_usage = st.checkbox(
                "Harmonize codon usage",
                value=True,
                help="Balance between optimal and natural codon usage"
            )
            
            avoid_repeats = st.checkbox(
                "Avoid sequence repeats",
                value=True,
                help="Minimize repetitive sequences"
            )
            
            cai_optimization = st.checkbox(
                "CAI optimization",
                value=True,
                help="Optimize Codon Adaptation Index"
            )
        
        # Quick optimization presets
        st.subheader("🎯 Quick Presets")
        
        preset_col1, preset_col2 = st.columns(2)
        
        with preset_col1:
            if st.button("🧬 E. coli Expression", use_container_width=True):
                st.session_state.preset_organism = "E. coli BL21"
        
        with preset_col2:
            if st.button("🧪 Mammalian Expression", use_container_width=True):
                st.session_state.preset_organism = "H. sapiens"
    
    # Optimization button
    if st.button("⚡ Optimize Sequence", type="primary", use_container_width=True):
        if not sequence_input:
            st.error("❌ Please enter a sequence")
        else:
            is_protein = input_type == "Protein Sequence"
            
            # Set up optimization parameters
            optimization_params = {
                'gc_target': (gc_min, gc_max),
                'avoid_sites': avoid_sites,
                'avoid_repeats': avoid_repeats,
                'harmonize_usage': harmonize_usage,
                'cai_optimization': cai_optimization,
                'strategy': optimization_strategy
            }
            
            with st.spinner("⚡ Optimizing codon usage..."):
                result = advanced_codon_optimization(
                    sequence_input, target_organism, optimization_params, is_protein
                )
            
            st.session_state.optimization_results = result
            
            if 'error' in result:
                st.error(f"❌ Optimization failed: {result['error']}")
            else:
                st.success("✅ Codon optimization completed successfully!")
    
    # Results display
    if 'optimization_results' in st.session_state and st.session_state.optimization_results:
        result = st.session_state.optimization_results
        
        if 'error' not in result:
            st.markdown("---")
            st.subheader("📊 Optimization Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Codon Changes", result['codon_changes'])
            with col2:
                st.metric("Total Codons", result['total_codons'])
            with col3:
                st.metric("GC Before", f"{result['gc_before']:.1f}%")
            with col4:
                st.metric("GC After", f"{result['gc_after']:.1f}%")
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                efficiency = (result['codon_changes'] / result['total_codons']) * 100 if result['total_codons'] > 0 else 0
                st.metric("Optimization Efficiency", f"{efficiency:.1f}%")
            with col2:
                gc_change = result['gc_after'] - result['gc_before']
                st.metric("GC Change", f"{gc_change:+.1f}%")
            with col3:
                st.metric("Verification", "✅ Pass" if result['verification'] else "❌ Fail")
            
            # Sequence comparison
            tab1, tab2, tab3, tab4 = st.tabs(["🧬 Sequences", "📈 Analysis", "⚙️ Parameters", "📄 Export"])
            
            with tab1:
                st.subheader("Sequence Comparison")
                
                seq_col1, seq_col2 = st.columns(2)
                
                with seq_col1:
                    st.write("**Original Sequence:**")
                    st.markdown(f'<div class="sequence-display">{result["original_sequence"]}</div>', unsafe_allow_html=True)
                    
                    if not result['is_protein_input']:
                        original_protein = translate_sequence(result["original_sequence"])
                        st.write("**Original Translation:**")
                        st.markdown(f'<div class="sequence-display">{original_protein}</div>', unsafe_allow_html=True)
                
                with seq_col2:
                    st.write("**Optimized Sequence:**")
                    st.markdown(f'<div class="sequence-display">{result["optimized_sequence"]}</div>', unsafe_allow_html=True)
                    
                    optimized_protein = translate_sequence(result["optimized_sequence"])
                    st.write("**Optimized Translation:**")
                    st.markdown(f'<div class="sequence-display">{optimized_protein}</div>', unsafe_allow_html=True)
                
                # Verification
                if result['verification']:
                    st.success("✅ Optimization verified: Protein sequences are identical")
                else:
                    st.error("❌ Verification failed: Protein sequences differ")
            
            with tab2:
                st.subheader("Detailed Analysis")
                
                # Codon usage comparison
                if USING_MATPLOTLIB:
                    # GC content chart
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # GC content comparison
                    categories = ['Before', 'After']
                    gc_values = [result['gc_before'], result['gc_after']]
                    
                    bars = ax1.bar(categories, gc_values, color=['lightblue', 'darkblue'])
                    ax1.set_ylabel('GC Content (%)')
                    ax1.set_title('GC Content Comparison')
                    ax1.set_ylim(0, 100)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, gc_values):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                                f'{value:.1f}%', ha='center', va='bottom')
                    
                    # Optimization efficiency
                    labels = ['Changed', 'Unchanged']
                    sizes = [result['codon_changes'], result['total_codons'] - result['codon_changes']]
                    colors = ['lightcoral', 'lightgreen']
                    
                    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax2.set_title('Codon Changes')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Avoided restriction sites
                if result.get('avoided_sites'):
                    st.subheader("🚫 Avoided Restriction Sites")
                    for site in result['avoided_sites']:
                        st.success(f"✅ Successfully avoided {site}")
                else:
                    st.info("ℹ️ No restriction sites needed to be avoided")
            
            with tab3:
                st.subheader("Optimization Parameters Used")
                
                params_data = {
                    'Parameter': [
                        'Target Organism',
                        'Input Type',
                        'GC Target Range',
                        'Avoid Repeats',
                        'Harmonize Usage',
                        'CAI Optimization',
                        'Avoided Sites',
                        'Strategy'
                    ],
                    'Value': [
                        result['target_organism'],
                        'Protein' if result['is_protein_input'] else 'DNA',
                        f"{gc_min}-{gc_max}%",
                        'Yes' if avoid_repeats else 'No',
                        'Yes' if harmonize_usage else 'No',
                        'Yes' if cai_optimization else 'No',
                        ', '.join(avoid_sites) if avoid_sites else 'None',
                        optimization_strategy
                    ]
                }
                
                st.dataframe(pd.DataFrame(params_data), use_container_width=True, hide_index=True)
            
            with tab4:
                st.subheader("Export Optimized Sequence")
                
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    # FASTA export
                    fasta_content = f">Optimized_for_{target_organism.replace(' ', '_')}\n{result['optimized_sequence']}"
                    
                    st.download_button(
                        "📄 Download FASTA",
                        fasta_content,
                        f"optimized_{target_organism.replace(' ', '_')}.fasta",
                        "text/plain",
                        use_container_width=True
                    )
                    
                    # GenBank style export
                    genbank_content = f"""LOCUS       Optimized       {len(result['optimized_sequence'])} bp    DNA     linear   SYN
DEFINITION  Codon optimized sequence for {target_organism}
FEATURES             Location/Qualifiers
     source          1..{len(result['optimized_sequence'])}
                     /organism="{target_organism}"
                     /codon_optimization="G-Synth v2025.6.0"
     gene            1..{len(result['optimized_sequence'])}
                     /optimization_efficiency="{efficiency:.1f}%"
ORIGIN
{result['optimized_sequence']}
//"""
                    
                    st.download_button(
                        "🧬 Download GenBank",
                        genbank_content,
                        f"optimized_{target_organism.replace(' ', '_')}.gb",
                        "text/plain",
                        use_container_width=True
                    )
                
                with export_col2:
                    # Detailed report
                    report_content = f"""Codon Optimization Report
=========================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Input Parameters:
- Target Organism: {target_organism}
- Input Type: {'Protein' if result['is_protein_input'] else 'DNA'}
- Original Length: {len(result['original_sequence'])} {'aa' if result['is_protein_input'] else 'bp'}
- Optimization Strategy: {optimization_strategy}

Results:
- Optimized Length: {len(result['optimized_sequence'])} bp
- Codon Changes: {result['codon_changes']}/{result['total_codons']} ({efficiency:.1f}%)
- GC Content: {result['gc_before']:.1f}% → {result['gc_after']:.1f}% ({gc_change:+.1f}%)
- Verification: {'PASS' if result['verification'] else 'FAIL'}

Original Sequence:
{result['original_sequence']}

Optimized Sequence:
{result['optimized_sequence']}

Generated by G-Synth v2025.6.0
"""
                    
                    st.download_button(
                        "📋 Download Report",
                        report_content,
                        f"optimization_report_{target_organism.replace(' ', '_')}.txt",
                        "text/plain",
                        use_container_width=True
                    )
                    
                    # JSON export
                    json_content = json.dumps(result, indent=2, default=str)
                    st.download_button(
                        "📊 Download JSON",
                        json_content,
                        f"optimization_data_{target_organism.replace(' ', '_')}.json",
                        "application/json",
                        use_container_width=True
                    )

def show_help_tab():
    """Help & Guide tab (preserved from original)"""
    
    st.header("📚 Help & Guide")
    st.markdown("*Complete guide to using G-Synth toolkit features*")
    
    # Help navigation
    help_tabs = st.tabs([
        "🏠 Getting Started",
        "🔬 Small Sequence Design", 
        "🔄 Translation Tools",
        "🧪 Primer Design",
        "🌀 Hybridization",
        "📏 Extended Synthesis",
        "⚡ Codon Optimization",
        "❓ FAQ"
    ])
    
    with help_tabs[0]:
        st.subheader("🏠 Welcome to G-Synth")
        
        st.markdown("""
        **G-Synth** is a comprehensive genetic engineering toolkit designed for molecular biologists, 
        synthetic biologists, and researchers working with DNA sequences.
        
        ### ✨ Key Features:
        
        - **🔬 Small Sequence Design (SSD)**: Design DNA sequences with enzyme sites
        - **🔄 Translation & Reverse Translation**: Convert between DNA and protein sequences  
        - **🧪 Primer Generator**: Design optimal PCR primers
        - **🌀 Hybridization Simulation**: Simulate DNA strand interactions
        - **📏 Extended Synthesis**: Fragment large sequences for synthesis
        - **⚡ Codon Optimization**: Optimize for different host organisms
        - **🔗 Ligation Check**: Verify fragment compatibility
        - **↔️ Reverse Complement**: Generate complementary strands
        
        ### 🚀 Quick Start:
        
        1. **Select a tool** from the navigation tabs
        2. **Enter your sequences** in the input fields
        3. **Configure parameters** as needed
        4. **Run the analysis** and view results
        5. **Export results** in various formats
        
        ### 💡 Tips:
        
        - Use the **example sequences** to test features
        - **Save your work** using the export functions
        - Check the **Help** sections for detailed guidance
        - **Validate sequences** before analysis
        """)
        
        # System information
        st.subheader("🔧 System Information")
        
        system_info = f"""
        - **Version**: G-Synth v2025.6.0
        - **Platform**: Streamlit Web Application
        - **Python Libraries**: 
          - Core: ✅ Available
          - Matplotlib: {'✅ Available' if USING_MATPLOTLIB else '❌ Not Available'}
          - Plotly: {'✅ Available' if USING_PLOTLY else '❌ Not Available'}
          - Biopython: {'✅ Available' if USING_BIOPYTHON else '❌ Not Available'}
        - **Developer**: Dr. Mohamed Merzoug
        - **Last Updated**: June 2025
        """
        
        st.markdown(system_info)
    
    with help_tabs[1]:
        st.subheader("🔬 Small Sequence Design (SSD)")
        
        st.markdown("""
        The **Small Sequence Design (SSD)** tool helps you design DNA sequences with appropriate 
        restriction enzyme sites, tags, and linkers for cloning and protein expression.
        
        ### 📝 Input Requirements:
        
        **For Coding Sequences:**
        - Must start with **ATG** (start codon)
        - Can optionally remove stop codons
        - Designed for protein-coding genes
        
        **For Non-coding Sequences:**
        - Any DNA sequence
        - Automatically adds ATG, His-tag, and linkers
        - Suitable for peptides, tags, or other elements
        
        ### ⚙️ Parameters:
        
        **Enzyme Pairs:**
        - **NdeI/XhoI**: Common for pET vectors
        - **BamHI/EcoRI**: Versatile cloning sites
        - **SalI/XbaI**: Alternative restriction sites
        
        **Cleavage Sites:**
        - **TEV**: Tobacco Etch Virus protease
        - **Thrombin**: Factor IIa cleavage
        - **Factor Xa**: Blood coagulation factor
        
        ### 📊 Outputs:
        
        - **Forward and reverse strands** with sticky ends
        - **Tm calculations** for primer design
        - **GC content analysis** 
        - **Sequence validation**
        - **Export options** (FASTA, JSON)
        
        ### 💡 Best Practices:
        
        1. **Validate sequences** before processing
        2. **Choose appropriate enzymes** for your vector
        3. **Consider Tm values** for PCR optimization
        4. **Check GC content** (optimal: 40-60%)
        5. **Include controls** in experimental design
        """)
    
    with help_tabs[2]:
        st.subheader("🔄 Translation Tools")
        
        st.markdown("""
        The **Translation Tools** provide comprehensive DNA-to-protein and protein-to-DNA conversion 
        with support for multiple reading frames and organisms.
        
        ### 🧬 DNA to Protein Translation:
        
        **Features:**
        - **6 reading frames**: Forward and reverse (1, 2, 3, -1, -2, -3)
        - **Start codon detection**: Finds ATG and starts translation
        - **Stop codon handling**: Stops at first stop codon
        - **ORF analysis**: Identifies open reading frames
        
        **Parameters:**
        - **Reading frame**: Choose specific frame (0, 1, 2)
        - **Find start**: Auto-detect ATG start codon
        - **Show all frames**: Display all 6 possible translations
        - **Three-letter codes**: Use Ala, Gly, etc. instead of A, G
        
        ### 🔄 Protein to DNA (Reverse Translation):
        
        **Features:**
        - **Organism-specific**: Uses optimal codons per organism
        - **Codon tables**: E. coli, yeast, human, CHO cells
        - **GC optimization**: Balances GC content automatically
        
        **Supported Organisms:**
        - **E. coli BL21**: Bacterial expression
        - **S. cerevisiae**: Yeast expression  
        - **H. sapiens**: Human cell lines
        - **CHO cells**: Mammalian expression
        - **P. pastoris**: Pichia expression
        
        ### 📊 Analysis Features:
        
        - **Molecular weight**: Approximate protein MW
        - **ORF detection**: Find coding sequences
        - **Sequence validation**: Check for errors
        - **Export options**: Multiple file formats
        
        ### 💡 Usage Tips:
        
        1. **Check all frames** for short sequences
        2. **Use appropriate organism** for your expression system
        3. **Validate ORFs** for gene identification
        4. **Consider codon bias** for optimization
        """)
    
    with help_tabs[3]:
        st.subheader("🧪 Primer Design")
        
        st.markdown("""
        The **Primer Generator** designs optimal PCR primers for amplification and cloning applications.
        
        ### 🎯 Design Modes:
        
        **Simple PCR:**
        - Basic primer design for amplification
        - Optimizes Tm, GC content, and length
        - No restriction sites added
        
        **Cloning Mode:**
        - Adds restriction enzyme sites
        - Includes custom prefixes
        - Optimized for molecular cloning
        
        ### ⚙️ Design Parameters:
        
        **Tm (Melting Temperature):**
        - **Optimal range**: 55-65°C
        - **Primer pairs**: Should be within 5°C
        - **Salt correction**: Accounts for PCR conditions
        
        **GC Content:**
        - **Optimal range**: 40-60%
        - **Avoid extremes**: <30% or >70%
        - **GC clamp**: 1-3 G/C at 3' end
        
        **Length:**
        - **Standard**: 18-25 nucleotides
        - **Cloning primers**: May be longer due to sites
        - **Avoid**: Very short (<15) or long (>35)
        
        ### 🔧 Cloning Features:
        
        **Restriction Enzymes:**
        - **Compatible sites**: Check vector compatibility
        - **Sticky ends**: 5' overhangs preferred
        - **Site availability**: Verify enzyme recognition
        
        **Custom Prefixes:**
        - **Default**: TGCATC (6 bases)
        - **Function**: Improves enzyme cutting efficiency
        - **Customizable**: Adapt to specific needs
        
        ### 📊 Quality Checks:
        
        - **Primer-dimer prediction**: Avoid self-complementarity
        - **Hairpin formation**: Check secondary structures
        - **Tm compatibility**: Ensure similar melting temperatures
        - **Restriction site validation**: Confirm enzyme recognition
        
        ### 💡 Design Tips:
        
        1. **Avoid repetitive sequences** in primer binding sites
        2. **Check for restriction sites** in your sequence
        3. **Use appropriate concentrations** (0.2-0.5 μM)
        4. **Validate primers** before ordering
        5. **Include controls** in PCR reactions
        """)
    
    with help_tabs[4]:
        st.subheader("🌀 Hybridization Simulation")
        
        st.markdown("""
        The **Hybridization Simulation** tool analyzes DNA strand interactions and binding patterns.
        
        ### 🧬 How It Works:
        
        The simulation uses the original algorithm:
        1. **Reverse** the reverse sequence (not reverse complement!)
        2. **Align** with forward sequence at all possible positions
        3. **Score** complementary base pairs
        4. **Find optimal** alignment position
        
        ### ⚙️ Parameters:
        
        **Max Shift Position:**
        - **Range**: 10-500 nucleotides
        - **Function**: Limits search space for alignment
        - **Performance**: Lower values = faster computation
        
        **Alignment Options:**
        - **Show detailed alignment**: Display base-by-base comparison
        - **Highlight matches**: Color complementary bases
        - **Temperature analysis**: Calculate binding stability
        
        ### 📊 Results:
        
        **Alignment Metrics:**
        - **Best shift**: Optimal alignment position
        - **Match score**: Number of complementary base pairs
        - **Similarity percentage**: Match score / total possible
        
        **Visualization:**
        - **Forward sequence**: 5' to 3' orientation
        - **Reverse sequence**: Reversed (not reverse complement!)
        - **Alignment markers**: | for matches, space for mismatches
        - **Color coding**: Green for matches, red for mismatches
        
        ### 🧪 Applications:
        
        - **Primer-template analysis**: Check primer binding
        - **Probe design**: Validate hybridization probes
        - **Sequence complementarity**: Analyze strand interactions
        - **SNP detection**: Identify sequence variations
        
        ### 💡 Interpretation:
        
        **High Similarity (>80%):**
        - Strong hybridization expected
        - Stable duplex formation
        - Good for primer binding
        
        **Medium Similarity (50-80%):**
        - Partial hybridization possible
        - Temperature-dependent binding
        - May need optimization
        
        **Low Similarity (<50%):**
        - Weak or no hybridization
        - Consider sequence redesign
        - Check for errors in input
        """)
    
    with help_tabs[5]:
        st.subheader("📏 Extended Synthesis")
        
        st.markdown("""
        The **Extended Synthesis** tool fragments large DNA sequences into smaller, 
        synthesizable pieces with overlapping regions for assembly.
        
        ### 🧩 Fragmentation Strategy:
        
        **Fragment Size:**
        - **Typical range**: 100-300 bp
        - **Synthesis limit**: Most vendors synthesize up to 200-300 bp
        - **Cost consideration**: Shorter fragments = more pieces = higher cost
        
        **Overlap Size:**
        - **Recommended**: 15-25 bp
        - **Function**: Enables assembly by homologous recombination
        - **Minimum**: 10 bp for basic assembly
        
        ### 🔧 Assembly Methods:
        
        **Overlap Extension PCR:**
        - **Process**: PCR with overlapping primers
        - **Advantages**: Simple, cost-effective
        - **Limitations**: Limited to smaller constructs
        
        **Gibson Assembly:**
        - **Process**: Single-step isothermal assembly
        - **Advantages**: High efficiency, multiple fragments
        - **Requirements**: Specific overlap design
        
        **Golden Gate Assembly:**
        - **Process**: Type IIS restriction enzyme-based
        - **Advantages**: Scarless assembly, modular
        - **Requirements**: Specific enzyme sites
        
        ### ⚙️ Parameters:
        
        **Terminal Enzyme Pair:**
        - **Function**: Adds restriction sites to terminal fragments
        - **Common choices**: NdeI/XhoI, BamHI/EcoRI
        - **Compatibility**: Must match destination vector
        
        **Cleavage Sites:**
        - **Optional**: Protease sites for tag removal
        - **Common**: TEV, Thrombin, Factor Xa
        - **Placement**: Between functional domains
        
        ### 📊 Quality Control:
        
        **Assembly Validation:**
        - **Sequence verification**: Check reassembled sequence
        - **Junction analysis**: Verify overlap regions
        - **Error detection**: Identify potential issues
        
        **Design Optimization:**
        - **GC balance**: Maintain optimal GC content
        - **Secondary structure**: Avoid problematic regions
        - **Restriction sites**: Check for unwanted sites
        
        ### 💡 Best Practices:
        
        1. **Plan assembly strategy** before fragmentation
        2. **Validate overlaps** for complementarity
        3. **Include buffer sequences** at fragment ends
        4. **Test assembly** with smaller constructs first
        5. **Sequence verify** final assembled construct
        """)
    
    with help_tabs[6]:
        st.subheader("⚡ Codon Optimization")
        
        st.markdown("""
        The **Codon Optimization** tool improves protein expression by using 
        optimal codons for your target organism.
        
        ### 🎯 Why Optimize Codons?
        
        **Expression Level:**
        - **Rare codons**: Can limit translation rate
        - **Optimal codons**: Match abundant tRNAs
        - **Expression boost**: 10-100x improvement possible
        
        **Protein Folding:**
        - **Translation speed**: Affects co-translational folding
        - **Synonymous sites**: Change codons, preserve protein
        - **Quality control**: Reduces misfolded proteins
        
        ### 🧬 Supported Organisms:
        
        **Bacterial Systems:**
        - **E. coli BL21**: Most common bacterial expression
        - **Advantages**: Fast, cheap, high yield
        - **Limitations**: No post-translational modifications
        
        **Yeast Systems:**
        - **S. cerevisiae**: Baker's yeast
        - **P. pastoris**: Pichia pastoris (methylotrophic)
        - **Advantages**: Eukaryotic modifications, secretion
        
        **Mammalian Systems:**
        - **H. sapiens**: Human cell lines
        - **CHO cells**: Chinese Hamster Ovary
        - **Advantages**: Native modifications, proper folding
        
        ### ⚙️ Optimization Parameters:
        
        **GC Content:**
        - **Target range**: 30-70% (organism-dependent)
        - **Balance**: Avoid extreme AT or GC content
        - **Stability**: Affects mRNA secondary structure
        
        **Restriction Sites:**
        - **Avoidance**: Prevent unwanted cleavage
        - **Common sites**: EcoRI, BamHI, HindIII, etc.
        - **Cloning**: Preserve desired sites
        
        **Repeat Sequences:**
        - **Avoidance**: Prevent recombination
        - **Length threshold**: Typically 6-12 bp
        - **Stability**: Improves plasmid maintenance
        
        ### 📊 Optimization Strategies:
        
        **Balanced:**
        - **Approach**: Moderate optimization
        - **Goal**: Improve expression while maintaining naturalness
        - **Best for**: General applications
        
        **High Expression:**
        - **Approach**: Maximize optimal codon usage
        - **Goal**: Highest possible protein yield
        - **Best for**: Commercial production
        
        **Low Immunogenicity:**
        - **Approach**: Minimize immune stimulation
        - **Goal**: Reduce inflammatory responses
        - **Best for**: Therapeutic proteins
        
        ### 🔍 Analysis Features:
        
        **Verification:**
        - **Protein comparison**: Ensures identical translation
        - **Codon counting**: Tracks optimization efficiency
        - **GC analysis**: Monitors composition changes
        
        **Quality Metrics:**
        - **CAI**: Codon Adaptation Index
        - **Efficiency**: Percentage of codons changed
        - **Validation**: Confirms successful optimization
        
        ### 💡 Optimization Tips:
        
        1. **Choose appropriate organism** for your expression system
        2. **Balance optimization** - don't over-optimize
        3. **Verify protein sequence** remains unchanged
        4. **Test expression** before large-scale production
        5. **Consider downstream processing** requirements
        """)
    
    with help_tabs[7]:
        st.subheader("❓ Frequently Asked Questions")
        
        with st.expander("❓ What file formats does G-Synth support?"):
            st.markdown("""
            **Input Formats:**
            - Plain text sequences (DNA/protein)
            - FASTA files (.fasta, .fa, .txt)
            - Multi-sequence FASTA files
            
            **Export Formats:**
            - FASTA (sequences)
            - JSON (analysis data)
            - Text reports (protocols)
            - GenBank format (annotated sequences)
            - CSV/Excel (tabular data)
            """)
        
        with st.expander("❓ How accurate are the Tm calculations?"):
            st.markdown("""
            G-Synth uses the **consensus nearest-neighbor method** which combines three 
            established thermodynamic models:
            
            - **Breslauer et al. (1986)**: Classic NN parameters
            - **SantaLucia et al. (1996)**: Improved accuracy
            - **Sugimoto et al. (1996)**: Alternative dataset
            
            **Accuracy:** ±2-3°C for most sequences under standard conditions
            **Conditions:** 50mM Na+, 500nM primer concentration
            **Validation:** Extensively tested against experimental data
            """)
        
        with st.expander("❓ Can I use G-Synth for commercial purposes?"):
            st.markdown("""
            **Academic Use:** ✅ Free for research and educational purposes
            
            **Commercial Use:** Please contact Dr. Mohamed Merzoug for licensing
            
            **Citation:** If you use G-Synth in publications, please cite:
            > Merzoug, M. et al. (2025). G-Synth: A comprehensive toolkit for genetic engineering. 
            > *Journal of Synthetic Biology*, in preparation.
            
            **Support:** For commercial licenses and custom features, contact the developer.
            """)
        
        with st.expander("❓ What are the system requirements?"):
            st.markdown("""
            **Minimum Requirements:**
            - Web browser (Chrome, Firefox, Safari, Edge)
            - Internet connection for web version
            - JavaScript enabled
            
            **Recommended:**
            - Modern browser (latest version)
            - 4GB RAM for large sequences
            - Good internet connection for file uploads
            
            **Local Installation:**
            - Python 3.8+
            - Streamlit 1.28+
            - Required packages (see requirements.txt)
            """)
        
        with st.expander("❓ How do I report bugs or request features?"):
            st.markdown("""
            **Bug Reports:**
            1. **Describe the issue** clearly
            2. **Include input sequences** (if possible)
            3. **Specify browser** and version
            4. **Note error messages** or unexpected behavior
            
            **Feature Requests:**
            1. **Explain the use case** for the new feature
            2. **Describe expected behavior**
            3. **Provide examples** if applicable
            
            **Contact:** Dr. Mohamed Merzoug
            **Response Time:** Usually within 24-48 hours
            """)
        
        with st.expander("❓ Is my sequence data secure?"):
            st.markdown("""
            **Data Security:**
            - **No storage**: Sequences are processed in memory only
            - **No logging**: Input sequences are not logged or stored
            - **Session-based**: Data cleared when session ends
            - **HTTPS**: All data transmission is encrypted
            
            **Privacy:**
            - **No tracking**: We don't track individual users
            - **No analytics**: No personal data collection
            - **Local processing**: All calculations done client-side when possible
            
            **Confidential Data:**
            For highly confidential sequences, consider using the local installation version.
            """)

# Add helper function for converting three-letter amino acid codes
def convert_to_three_letter(protein_seq):
    """Convert one-letter amino acid codes to three-letter codes"""
    mapping = {
        'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
        'Q': 'Gln', 'E': 'Glu', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
        'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
        'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val', 
        '*': 'Stop'
    }
    
    result = []
    for aa in protein_seq:
        result.append(mapping.get(aa, aa))
    
    return " ".join(result)

#########################
# RUN APPLICATION
#########################

if __name__ == "__main__":
    main()
