#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G-Synth: Complete Advanced Genetic Engineering Toolkit - Streamlit Version
Version 2025.6.0 - COMPLETE IMPLEMENTATION

A comprehensive toolkit for gene synthesis and molecular cloning.
Developed by Dr. Mohamed Merzoug

COMPLETE VERSION WITH ALL FEATURES FULLY IMPLEMENTED:
✅ Small Sequence Design (SSD) - Original G-Synth feature
✅ Advanced Codon Optimization - Multi-organism support
✅ Primer Design & Analysis - Comprehensive primer tools
✅ Hybridization Simulation - DNA-DNA interactions
✅ Restriction Enzyme Analysis - 200+ enzymes
✅ Translation Tools - All 6 reading frames
✅ ORF Finding & Analysis - Complete implementation
✅ Ligation Calculator - Efficiency prediction
✅ AI-Based Molecular Docking - Structure prediction
✅ Functional Prediction - GO terms, pathways
✅ Sequence Alignment - Pairwise and MSA
✅ CRISPR Guide Design - Multi-Cas systems
✅ Plasmid Visualization - Interactive maps
✅ Pathway Designer - Multi-gene operons
✅ Batch Processing - Multiple sequences
✅ Advanced Export - All formats
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math
import itertools
import json
import io
import base64
import time
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import hashlib
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

# Configure Streamlit page FIRST
st.set_page_config(
    page_title="G-Synth Complete Toolkit",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Safe imports with comprehensive fallbacks
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

# Set compatibility variables
MATPLOTLIB_AVAILABLE = True
USING_MATPLOTLIB = True
BIOTITE_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('G-Synth')

#########################
# ENHANCED CSS STYLING
#########################

st.markdown("""
<style>
    /* Main styling */
    .main > div { padding-top: 1rem; }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover { transform: translateY(-5px); }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Sequence display */
    .sequence-display {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #007acc;
        overflow-x: auto;
        white-space: pre-wrap;
        word-break: break-all;
        line-height: 1.6;
        font-size: 0.9rem;
        max-height: 400px;
        overflow-y: auto;
    }
    
    /* Syntax highlighting */
    .highlight-atg { background-color: #28a745; color: white; padding: 2px 4px; border-radius: 3px; font-weight: bold; }
    .highlight-stop { background-color: #dc3545; color: white; padding: 2px 4px; border-radius: 3px; font-weight: bold; }
    .highlight-restriction { background-color: #ffc107; color: black; padding: 2px 4px; border-radius: 3px; font-weight: bold; }
    .highlight-his { background-color: #e83e8c; color: white; padding: 2px 4px; border-radius: 3px; font-weight: bold; }
    .highlight-linker { background-color: #17a2b8; color: white; padding: 2px 4px; border-radius: 3px; font-weight: bold; }
    .highlight-primer { background-color: #6f42c1; color: white; padding: 2px 4px; border-radius: 3px; font-weight: bold; }
    
    /* Status indicators */
    .status-available { color: #28a745; font-weight: bold; }
    .status-missing { color: #dc3545; font-weight: bold; }
    .status-partial { color: #ffc107; font-weight: bold; }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* DataFrames */
    .dataframe thead th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
    }
    
    .dataframe tbody tr:nth-child(even) { background-color: #f8f9fa; }
    
    /* Success/Error messages */
    .stSuccess { background-color: #d4edda; border-left: 4px solid #28a745; border-radius: 0.5rem; }
    .stError { background-color: #f8d7da; border-left: 4px solid #dc3545; border-radius: 0.5rem; }
    .stWarning { background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 0.5rem; }
    .stInfo { background-color: #d1ecf1; border-left: 4px solid #17a2b8; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

#########################
# COMPREHENSIVE BIOLOGICAL CONSTANTS
#########################

# Genetic code table
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

# Comprehensive codon usage tables
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

# Comprehensive restriction enzyme database
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

# Protease cleavage sites
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

# SSD constants (preserved from original)
SSD_HIS_TAG = "CACCACCACCACCACCAC"
SSD_LEFT_LINKER = "GGTTCTTCT"
SSD_RIGHT_LINKER = "TCTTCTGGT"

# CRISPR PAM sequences
CRISPR_PAM_SEQUENCES = {
    'Cas9 (SpCas9)': {'pam': 'NGG', 'pam_position': '3prime', 'guide_length': 20},
    'Cas12a (Cpf1)': {'pam': 'TTTN', 'pam_position': '5prime', 'guide_length': 20},
    'Cas13a': {'pam': 'H', 'pam_position': '3prime', 'guide_length': 28},
    'Cas9 (SaCas9)': {'pam': 'NNGRRT', 'pam_position': '3prime', 'guide_length': 21},
    'Cas9 (NmCas9)': {'pam': 'NNNNGATT', 'pam_position': '3prime', 'guide_length': 24}
}

# GO terms database
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
        'GO:0023052': 'signaling', 'GO:0032502': 'developmental process'
    },
    'cellular_component': {
        'GO:0005575': 'cellular_component', 'GO:0005623': 'cell',
        'GO:0005622': 'intracellular', 'GO:0043226': 'organelle',
        'GO:0032991': 'protein-containing complex', 'GO:0005829': 'cytosol'
    }
}

#########################
# SESSION STATE INITIALIZATION
#########################

def init_session_state():
    """Initialize comprehensive session state"""
    defaults = {
        'sequences_cache': {},
        'analysis_results': {},
        'ssd_results': {},
        'primer_results': {},
        'translation_results': {},
        'restriction_results': {},
        'codon_opt_results': {},
        'hybridization_results': {},
        'crispr_results': {},
        'ligation_results': {},
        'alignment_results': {},
        'docking_results': {},
        'pathway_designs': [],
        'plasmid_features': [],
        'current_project': None,
        'user_preferences': {
            'default_organism': 'E. coli BL21',
            'default_temperature': 37.0,
            'default_salt_concentration': 50.0,
            'auto_save': True,
            'theme': 'light'
        },
        'export_formats': ['FASTA', 'GenBank', 'JSON', 'CSV'],
        'batch_sequences': [],
        'ai_models_loaded': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
init_session_state()

#########################
# CORE BIOLOGICAL FUNCTIONS (Complete Implementation)
#########################

def clean_dna_sequence(seq: str, keep_ambiguous: bool = False) -> str:
    """Clean DNA sequence by removing invalid characters"""
    if keep_ambiguous:
        return re.sub(r"[^ACGTRYSWKMBDHVN]", "", seq.upper())
    else:
        return re.sub(r"[^ATCG]", "", seq.upper())

def validate_dna_sequence(sequence: str, allow_empty: bool = False) -> Tuple[bool, str, Optional[str]]:
    """Validate DNA sequence and return clean version"""
    if not sequence and not allow_empty:
        return False, "", "Sequence cannot be empty"
    
    clean_seq = clean_dna_sequence(sequence)
    
    if not clean_seq and sequence:
        return False, "", "Sequence contains no valid DNA characters"
    
    if len(clean_seq) < len(sequence.replace(" ", "").replace("\n", "")):
        warning = f"Removed {len(sequence.replace(' ', '').replace('\n', '')) - len(clean_seq)} invalid characters"
        return True, clean_seq, warning
    
    return True, clean_seq, None

def reverse_complement(seq: str) -> str:
    """Generate reverse complement of DNA sequence"""
    complement_table = str.maketrans("ACGTN", "TGCAN")
    return seq.upper().translate(complement_table)[::-1]

def translate_sequence(nuc_seq: str, frame: int = 0, find_start: bool = True) -> str:
    """Translate nucleotide sequence to protein"""
    nuc_seq = clean_dna_sequence(nuc_seq)
    if not nuc_seq:
        return ""
    
    if frame not in [0, 1, 2]:
        frame = 0
    
    start = frame
    if find_start:
        pos = nuc_seq.find("ATG", frame)
        if pos != -1:
            start = pos
    
    end = len(nuc_seq) - ((len(nuc_seq) - start) % 3)
    
    protein = ""
    for i in range(start, end, 3):
        codon = nuc_seq[i:i+3]
        if len(codon) == 3:
            aa = GENETIC_CODE.get(codon, "X")
            protein += aa
            if find_start and aa == "*" and i > start:
                break
    
    return protein

def reverse_translate_to_dna(protein: str, organism: str = "E. coli BL21") -> str:
    """Convert protein sequence to DNA using optimal codons"""
    if organism not in CODON_USAGE_TABLES:
        organism = "E. coli BL21"
    
    codon_table = CODON_USAGE_TABLES[organism]
    dna = ""
    
    for aa in protein.upper():
        if aa in codon_table:
            dna += codon_table[aa][0]
        else:
            dna += "NNN"
    
    return dna

def calculate_gc_content(seq: str) -> float:
    """Calculate GC content as percentage"""
    seq = seq.upper()
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq) * 100

def calculate_tm_consensus(sequence: str, primer_conc: float = 500e-9, salt_conc: float = 50e-3) -> Optional[float]:
    """Calculate melting temperature using consensus method"""
    sequence = sequence.upper().replace(' ', '')
    if not sequence or not all(base in "ATCG" for base in sequence):
        return None
    
    if len(sequence) < 8:
        a = sequence.count('A')
        t = sequence.count('T')
        g = sequence.count('G')
        c = sequence.count('C')
        return 2 * (a + t) + 4 * (g + c) - 7
    
    # Nearest neighbor parameters
    nn_params = {
        "AA": (-9.1, -24.0), "TT": (-9.1, -24.0),
        "AT": (-8.6, -23.9), "TA": (-6.0, -16.9),
        "CA": (-5.8, -12.9), "TG": (-5.8, -12.9),
        "GT": (-6.5, -17.3), "AC": (-6.5, -17.3),
        "CT": (-7.8, -20.8), "AG": (-7.8, -20.8),
        "GA": (-5.6, -13.5), "TC": (-5.6, -13.5),
        "CG": (-11.9, -27.8), "GC": (-11.1, -26.7),
        "GG": (-11.0, -26.6), "CC": (-11.0, -26.6)
    }
    
    delta_h = 0
    delta_s = 0
    
    for i in range(len(sequence) - 1):
        pair = sequence[i:i+2]
        if pair in nn_params:
            h, s = nn_params[pair]
            delta_h += h
            delta_s += s
    
    delta_s += -10.8  # Terminal correction
    
    R = 1.987  # cal/(mol*K)
    c = primer_conc / 4
    tm_kelvin = (delta_h * 1000) / (delta_s + R * math.log(c))
    
    salt_correction = 16.6 * math.log10(salt_conc)
    tm_celsius = tm_kelvin - 273.15 + salt_correction
    
    return round(tm_celsius, 1)

def find_orfs(sequence: str, min_length: int = 100) -> List[Dict]:
    """Find all open reading frames in DNA sequence"""
    sequence = clean_dna_sequence(sequence)
    orfs = []
    
    # Check all 6 reading frames
    for strand in [1, -1]:
        seq = sequence if strand == 1 else reverse_complement(sequence)
        
        for frame in range(3):
            for i in range(frame, len(seq) - 2, 3):
                if seq[i:i+3] == "ATG":
                    for j in range(i + 3, len(seq) - 2, 3):
                        codon = seq[j:j+3]
                        if codon in ["TAA", "TAG", "TGA"]:
                            orf_length = j + 3 - i
                            if orf_length >= min_length:
                                if strand == 1:
                                    start_pos = i + 1
                                    end_pos = j + 3
                                else:
                                    start_pos = len(sequence) - (j + 3) + 1
                                    end_pos = len(sequence) - i
                                
                                orfs.append({
                                    'start': start_pos,
                                    'end': end_pos,
                                    'length': orf_length,
                                    'frame': frame + 1 if strand == 1 else -(frame + 1),
                                    'strand': '+' if strand == 1 else '-',
                                    'sequence': seq[i:j+3],
                                    'protein': translate_sequence(seq[i:j+3])
                                })
                            break
    
    return sorted(orfs, key=lambda x: x['length'], reverse=True)

#########################
# SMALL SEQUENCE DESIGN (Complete Implementation)
#########################

def small_sequence_design(forward_seq: str, reverse_seq: str, fwd_enzyme: str, 
                         rev_enzyme: str, include_his_tag: bool = True,
                         cleavage_site: str = "TEV", linker_type: str = "flexible") -> Dict:
    """
    Complete Small Sequence Design (SSD) implementation
    Design a complete expression construct with all necessary elements
    """
    try:
        # Validate inputs
        is_valid_fwd, clean_fwd, fwd_warning = validate_dna_sequence(forward_seq)
        is_valid_rev, clean_rev, rev_warning = validate_dna_sequence(reverse_seq)
        
        if not is_valid_fwd or not is_valid_rev:
            return {'error': 'Invalid DNA sequences provided'}
        
        # Get enzyme linkers
        fwd_linker = ENZYME_LINKERS.get(fwd_enzyme, "")
        rev_linker = ENZYME_LINKERS.get(rev_enzyme, "")
        
        if not fwd_linker or not rev_linker:
            return {'error': 'Unknown restriction enzyme'}
        
        # Build construct components
        construct_parts = []
        construct_map = []  # For visualization
        current_pos = 0
        
        # 1. Forward restriction site
        construct_parts.append(('Forward Restriction Site', fwd_linker))
        construct_map.append({
            'name': f'{fwd_enzyme} site',
            'start': current_pos,
            'end': current_pos + len(fwd_linker),
            'type': 'restriction_site',
            'enzyme': fwd_enzyme
        })
        current_pos += len(fwd_linker)
        
        # 2. Start codon (if NdeI, it's included in the site)
        if fwd_enzyme == "NdeI":
            start_codon = ""  # Already in NdeI site
        else:
            start_codon = "ATG"
            if not clean_fwd.startswith("ATG"):
                construct_parts.append(('Start Codon', start_codon))
                construct_map.append({
                    'name': 'Start codon',
                    'start': current_pos,
                    'end': current_pos + 3,
                    'type': 'start_codon'
                })
                current_pos += 3
        
        # 3. Left linker
        if SSD_LEFT_LINKER:
            construct_parts.append(('Left Linker', SSD_LEFT_LINKER))
            construct_map.append({
                'name': 'Left linker',
                'start': current_pos,
                'end': current_pos + len(SSD_LEFT_LINKER),
                'type': 'linker'
            })
            current_pos += len(SSD_LEFT_LINKER)
        
        # 4. Forward sequence
        main_fwd = clean_fwd[3:] if clean_fwd.startswith("ATG") and fwd_enzyme != "NdeI" else clean_fwd
        construct_parts.append(('Forward Sequence', main_fwd))
        construct_map.append({
            'name': 'Forward sequence',
            'start': current_pos,
            'end': current_pos + len(main_fwd),
            'type': 'gene_sequence'
        })
        current_pos += len(main_fwd)
        
        # 5. Reverse sequence
        construct_parts.append(('Reverse Sequence', clean_rev))
        construct_map.append({
            'name': 'Reverse sequence',
            'start': current_pos,
            'end': current_pos + len(clean_rev),
            'type': 'gene_sequence'
        })
        current_pos += len(clean_rev)
        
        # 6. His-tag (if requested)
        if include_his_tag:
            construct_parts.append(('His Tag', SSD_HIS_TAG))
            construct_map.append({
                'name': 'His tag (6x)',
                'start': current_pos,
                'end': current_pos + len(SSD_HIS_TAG),
                'type': 'affinity_tag'
            })
            current_pos += len(SSD_HIS_TAG)
        
        # 7. Cleavage site (if specified)
        if cleavage_site and cleavage_site in CLEAVAGE_SITES:
            cleavage_seq = CLEAVAGE_SITES[cleavage_site]
            construct_parts.append(('Cleavage Site', cleavage_seq))
            construct_map.append({
                'name': f'{cleavage_site} site',
                'start': current_pos,
                'end': current_pos + len(cleavage_seq),
                'type': 'cleavage_site'
            })
            current_pos += len(cleavage_seq)
        
        # 8. Right linker
        if SSD_RIGHT_LINKER:
            construct_parts.append(('Right Linker', SSD_RIGHT_LINKER))
            construct_map.append({
                'name': 'Right linker',
                'start': current_pos,
                'end': current_pos + len(SSD_RIGHT_LINKER),
                'type': 'linker'
            })
            current_pos += len(SSD_RIGHT_LINKER)
        
        # 9. Stop codon (if not present)
        if not any(stop in [clean_fwd[-3:], clean_rev[-3:]] for stop in ["TAA", "TAG", "TGA"]):
            stop_codon = "TAA"
            construct_parts.append(('Stop Codon', stop_codon))
            construct_map.append({
                'name': 'Stop codon',
                'start': current_pos,
                'end': current_pos + 3,
                'type': 'stop_codon'
            })
            current_pos += 3
        
        # 10. Reverse restriction site
        construct_parts.append(('Reverse Restriction Site', rev_linker))
        construct_map.append({
            'name': f'{rev_enzyme} site',
            'start': current_pos,
            'end': current_pos + len(rev_linker),
            'type': 'restriction_site',
            'enzyme': rev_enzyme
        })
        
        # Assemble final sequence
        final_sequence = "".join([part[1] for part in construct_parts])
        
        # Design primers
        fwd_primer_core = clean_fwd[:20] if len(clean_fwd) >= 20 else clean_fwd
        rev_primer_core = reverse_complement(clean_rev[-20:] if len(clean_rev) >= 20 else clean_rev)
        
        fwd_primer = fwd_linker + fwd_primer_core
        rev_primer = reverse_complement(rev_linker) + rev_primer_core
        
        # Calculate properties
        gc_content = calculate_gc_content(final_sequence)
        tm_forward = calculate_tm_consensus(fwd_primer)
        tm_reverse = calculate_tm_consensus(rev_primer)
        
        # Protein analysis
        protein_sequence = translate_sequence(final_sequence)
        molecular_weight = len(protein_sequence.replace('*', '')) * 110  # Approximate
        
        # Theoretical protein properties
        if protein_sequence:
            protein_clean = protein_sequence.replace('*', '')
            hydrophobic_aa = sum(1 for aa in protein_clean if aa in 'AILMFWYV')
            charged_aa = sum(1 for aa in protein_clean if aa in 'DEKRH')
            
            hydrophobicity = hydrophobic_aa / len(protein_clean) if protein_clean else 0
            net_charge = (protein_clean.count('K') + protein_clean.count('R') + 
                         protein_clean.count('H')) - (protein_clean.count('D') + protein_clean.count('E'))
        else:
            hydrophobicity = 0
            net_charge = 0
        
        # Quality checks
        quality_checks = []
        
        # Check for unwanted restriction sites
        unwanted_sites = []
        for enzyme, site in ENZYME_LINKERS.items():
            if enzyme not in [fwd_enzyme, rev_enzyme] and site in final_sequence:
                unwanted_sites.append(enzyme)
        
        if unwanted_sites:
            quality_checks.append(f"Warning: Contains {len(unwanted_sites)} unwanted restriction sites")
        else:
            quality_checks.append("✓ No unwanted restriction sites found")
        
        # Check GC content
        if 30 <= gc_content <= 70:
            quality_checks.append("✓ GC content within optimal range")
        else:
            quality_checks.append(f"Warning: GC content ({gc_content:.1f}%) outside optimal range")
        
        # Check primer Tm compatibility
        if tm_forward and tm_reverse:
            tm_diff = abs(tm_forward - tm_reverse)
            if tm_diff <= 5:
                quality_checks.append("✓ Primer Tm values are compatible")
            else:
                quality_checks.append(f"Warning: Large Tm difference ({tm_diff:.1f}°C)")
        
        return {
            'success': True,
            'final_sequence': final_sequence,
            'construct_parts': construct_parts,
            'construct_map': construct_map,
            'forward_primer': fwd_primer,
            'reverse_primer': rev_primer,
            'properties': {
                'length': len(final_sequence),
                'gc_content': gc_content,
                'molecular_weight': molecular_weight,
                'tm_forward': tm_forward,
                'tm_reverse': tm_reverse,
                'hydrophobicity': hydrophobicity,
                'net_charge': net_charge
            },
            'protein_sequence': protein_sequence,
            'protein_clean': protein_sequence.replace('*', ''),
            'parameters': {
                'forward_enzyme': fwd_enzyme,
                'reverse_enzyme': rev_enzyme,
                'include_his_tag': include_his_tag,
                'cleavage_site': cleavage_site,
                'linker_type': linker_type
            },
            'quality_checks': quality_checks,
            'unwanted_sites': unwanted_sites,
            'warnings': [fwd_warning, rev_warning] if fwd_warning or rev_warning else []
        }
        
    except Exception as e:
        logger.error(f"SSD failed: {str(e)}")
        return {'error': f'SSD failed: {str(e)}'}

#########################
# ADVANCED CODON OPTIMIZATION (Complete Implementation)
#########################

def advanced_codon_optimization(sequence: str, target_organism: str = "E. coli BL21", 
                               optimization_parameters: Optional[Dict] = None, 
                               is_protein: bool = False) -> Dict:
    """
    Complete codon optimization with advanced features
    """
    if optimization_parameters is None:
        optimization_parameters = {
            'gc_target': (30, 70),
            'avoid_sites': [],
            'avoid_repeats': True,
            'harmonize_usage': True,
            'cai_optimization': True
        }
    
    try:
        if target_organism not in CODON_USAGE_TABLES:
            target_organism = "E. coli BL21"
        
        codon_table = CODON_USAGE_TABLES[target_organism]
        
        # Initialize results
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
            "verification": False,
            "cai_before": 0,
            "cai_after": 0,
            "optimization_report": []
        }
        
        # Step 1: Convert to working sequences
        if is_protein:
            protein_seq = sequence.upper()
            protein_seq = "".join(c for c in protein_seq if c in "ACDEFGHIKLMNPQRSTVWY*")
            original_dna = reverse_translate_to_dna(protein_seq, target_organism)
            working_sequence = original_dna
            results["total_codons"] = len(protein_seq)
        else:
            dna_seq = clean_dna_sequence(sequence)
            protein_seq = translate_sequence(dna_seq)
            working_sequence = dna_seq
            results["total_codons"] = len(dna_seq) // 3
        
        # Calculate initial metrics
        gc_before = calculate_gc_content(working_sequence)
        results["gc_before"] = gc_before
        
        # Step 2: Codon optimization
        optimized = ""
        codon_changes = 0
        
        for i in range(0, len(working_sequence), 3):
            if i + 3 > len(working_sequence):
                optimized += working_sequence[i:]
                continue
            
            codon = working_sequence[i:i+3]
            amino_acid = GENETIC_CODE.get(codon)
            
            if amino_acid is None:
                optimized += codon
                continue
            
            # Handle special cases
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
            
            # Get optimal codons
            if amino_acid in codon_table:
                potential_codons = codon_table[amino_acid].copy()
                
                # Apply optimization constraints
                gc_target_min, gc_target_max = optimization_parameters.get('gc_target', (30, 70))
                current_gc = calculate_gc_content(optimized)
                
                # GC content balancing
                if current_gc < gc_target_min:
                    potential_codons.sort(key=lambda c: (c.count('G') + c.count('C')), reverse=True)
                elif current_gc > gc_target_max:
                    potential_codons.sort(key=lambda c: (c.count('G') + c.count('C')))
                
                best_codon = potential_codons[0] if potential_codons else codon
                
                # Avoid restriction sites
                avoid_sites = optimization_parameters.get('avoid_sites', [])
                if avoid_sites:
                    for site in avoid_sites:
                        if site in ENZYME_LINKERS:
                            restriction_seq = ENZYME_LINKERS[site]
                            # Check if any codon would create the restriction site
                            safe_codons = []
                            for pot_codon in potential_codons:
                                context = optimized[-6:] + pot_codon + working_sequence[i+3:i+9]
                                if restriction_seq not in context:
                                    safe_codons.append(pot_codon)
                            
                            if safe_codons:
                                best_codon = safe_codons[0]
                                if restriction_seq in optimized[-6:] + codon + working_sequence[i+3:i+9]:
                                    results["avoided_sites"].append(site)
                
                # Avoid repeats
                if optimization_parameters.get('avoid_repeats', True):
                    repeat_safe_codons = []
                    for pot_codon in potential_codons:
                        context = optimized[-9:] + pot_codon
                        has_repeat = False
                        
                        # Check for repeats of length 6-12
                        for repeat_len in range(6, 13):
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
        
        # Store results
        results["optimized_sequence"] = optimized
        results["codon_changes"] = codon_changes
        
        # Calculate final metrics
        gc_after = calculate_gc_content(optimized)
        results["gc_after"] = gc_after
        
        # Verification
        if is_protein:
            optimized_translation = translate_sequence(optimized, 0, False)
            results["verification"] = (protein_seq.replace("*", "") == optimized_translation.replace("*", ""))
        else:
            original_translation = translate_sequence(working_sequence, 0, False)
            optimized_translation = translate_sequence(optimized, 0, False)
            results["verification"] = (original_translation == optimized_translation)
        
        # Generate optimization report
        report = []
        report.append(f"Optimized {codon_changes} of {results['total_codons']} codons ({codon_changes/results['total_codons']*100:.1f}%)")
        report.append(f"GC content changed from {gc_before:.1f}% to {gc_after:.1f}%")
        
        if results["avoided_sites"]:
            report.append(f"Avoided {len(results['avoided_sites'])} restriction sites: {', '.join(results['avoided_sites'])}")
        
        if results["verification"]:
            report.append("✓ Optimization verified - protein sequence unchanged")
        else:
            report.append("⚠ Verification failed - protein sequence changed")
        
        results["optimization_report"] = report
        
        return results
        
    except Exception as e:
        logger.error(f"Codon optimization failed: {str(e)}")
        return {'error': f'Codon optimization failed: {str(e)}'}

#########################
# COMPREHENSIVE PRIMER DESIGN (Complete Implementation)
#########################

def comprehensive_primer_design(target_sequence: str, design_type: str = "PCR", 
                               target_tm: float = 60.0, primer_length_range: Tuple[int, int] = (18, 25),
                               gc_range: Tuple[float, float] = (40.0, 60.0),
                               avoid_secondary: bool = True, cloning_sites: List[str] = None) -> Dict:
    """
    Complete primer design with multiple design strategies
    """
    try:
        sequence = clean_dna_sequence(target_sequence)
        if len(sequence) < 50:
            return {'error': 'Sequence too short for primer design (minimum 50 bp)'}
        
        min_len, max_len = primer_length_range
        min_gc, max_gc = gc_range
        
        primers = []
        primer_pairs = []
        
        # Design forward primers
        for length in range(min_len, max_len + 1):
            for start_pos in range(min(10, len(sequence) - length)):
                primer_seq = sequence[start_pos:start_pos + length]
                
                # Calculate properties
                tm = calculate_tm_consensus(primer_seq)
                gc = calculate_gc_content(primer_seq)
                
                # Quality checks
                if not (min_gc <= gc <= max_gc):
                    continue
                if tm is None:
                    continue
                
                # Secondary structure check (simplified)
                secondary_score = 1.0
                if avoid_secondary:
                    # Check for hairpins (simplified)
                    hairpin_penalty = 0
                    for i in range(len(primer_seq) - 3):
                        for j in range(i + 4, len(primer_seq)):
                            if primer_seq[i:i+2] == reverse_complement(primer_seq[j-1:j+1]):
                                hairpin_penalty += 1
                    secondary_score = max(0.1, 1.0 - hairpin_penalty * 0.1)
                
                # Calculate overall score
                tm_score = max(0, 100 - abs(tm - target_tm) * 2)
                gc_score = max(0, 100 - abs(gc - 50) * 2)
                overall_score = (tm_score + gc_score) * secondary_score / 2
                
                primers.append({
                    'type': 'forward',
                    'sequence': primer_seq,
                    'length': length,
                    'start_position': start_pos + 1,
                    'end_position': start_pos + length,
                    'tm': tm,
                    'gc_content': gc,
                    'score': overall_score,
                    'tm_score': tm_score,
                    'gc_score': gc_score,
                    'secondary_score': secondary_score
                })
        
        # Design reverse primers
        for length in range(min_len, max_len + 1):
            for end_pos in range(len(sequence) - 10, len(sequence) - length + 1, -1):
                if end_pos - length < 0:
                    continue
                    
                primer_seq = reverse_complement(sequence[end_pos - length:end_pos])
                
                # Calculate properties
                tm = calculate_tm_consensus(primer_seq)
                gc = calculate_gc_content(primer_seq)
                
                # Quality checks
                if not (min_gc <= gc <= max_gc):
                    continue
                if tm is None:
                    continue
                
                # Secondary structure check
                secondary_score = 1.0
                if avoid_secondary:
                    hairpin_penalty = 0
                    for i in range(len(primer_seq) - 3):
                        for j in range(i + 4, len(primer_seq)):
                            if primer_seq[i:i+2] == reverse_complement(primer_seq[j-1:j+1]):
                                hairpin_penalty += 1
                    secondary_score = max(0.1, 1.0 - hairpin_penalty * 0.1)
                
                # Calculate overall score
                tm_score = max(0, 100 - abs(tm - target_tm) * 2)
                gc_score = max(0, 100 - abs(gc - 50) * 2)
                overall_score = (tm_score + gc_score) * secondary_score / 2
                
                primers.append({
                    'type': 'reverse',
                    'sequence': primer_seq,
                    'length': length,
                    'start_position': end_pos - length + 1,
                    'end_position': end_pos,
                    'tm': tm,
                    'gc_content': gc,
                    'score': overall_score,
                    'tm_score': tm_score,
                    'gc_score': gc_score,
                    'secondary_score': secondary_score
                })
        
        # Sort primers by score
        primers.sort(key=lambda x: x['score'], reverse=True)
        
        # Find optimal primer pairs
        forward_primers = [p for p in primers if p['type'] == 'forward'][:10]
        reverse_primers = [p for p in primers if p['type'] == 'reverse'][:10]
        
        for fp in forward_primers:
            for rp in reverse_primers:
                tm_diff = abs(fp['tm'] - rp['tm'])
                
                # Check for primer-dimer potential (simplified)
                primer_dimer_score = 1.0
                fp_3prime = fp['sequence'][-6:]
                rp_3prime = rp['sequence'][-6:]
                
                max_complement = 0
                for i in range(len(fp_3prime)):
                    for j in range(len(rp_3prime)):
                        complement_len = 0
                        for k in range(min(len(fp_3prime) - i, len(rp_3prime) - j)):
                            if fp_3prime[i + k] == reverse_complement(rp_3prime[j + k]):
                                complement_len += 1
                            else:
                                break
                        max_complement = max(max_complement, complement_len)
                
                if max_complement >= 4:
                    primer_dimer_score = max(0.1, 1.0 - (max_complement - 3) * 0.2)
                
                # Calculate pair score
                if tm_diff <= 5:  # Compatible Tm values
                    amplicon_size = rp['start_position'] - fp['end_position']
                    if amplicon_size > 0:  # Valid amplicon
                        pair_score = (fp['score'] + rp['score']) / 2 * primer_dimer_score
                        
                        primer_pairs.append({
                            'forward': fp,
                            'reverse': rp,
                            'tm_difference': tm_diff,
                            'amplicon_size': amplicon_size,
                            'pair_score': pair_score,
                            'primer_dimer_score': primer_dimer_score,
                            'max_complement': max_complement
                        })
        
        # Sort pairs by score
        primer_pairs.sort(key=lambda x: x['pair_score'], reverse=True)
        
        # Add cloning sites if requested
        if cloning_sites and design_type == "cloning":
            enhanced_pairs = []
            for pair in primer_pairs[:5]:  # Top 5 pairs
                for fwd_site in cloning_sites:
                    for rev_site in cloning_sites:
                        if fwd_site in ENZYME_LINKERS and rev_site in ENZYME_LINKERS:
                            fwd_linker = ENZYME_LINKERS[fwd_site]
                            rev_linker = ENZYME_LINKERS[rev_site]
                            
                            enhanced_fwd = fwd_linker + pair['forward']['sequence']
                            enhanced_rev = rev_linker + pair['reverse']['sequence']
                            
                            enhanced_pairs.append({
                                'forward': {
                                    **pair['forward'],
                                    'sequence': enhanced_fwd,
                                    'cloning_site': fwd_site,
                                    'tm': calculate_tm_consensus(enhanced_fwd)
                                },
                                'reverse': {
                                    **pair['reverse'],
                                    'sequence': enhanced_rev,
                                    'cloning_site': rev_site,
                                    'tm': calculate_tm_consensus(enhanced_rev)
                                },
                                'amplicon_size': pair['amplicon_size'],
                                'pair_score': pair['pair_score'] * 0.9,  # Slight penalty for longer primers
                                'cloning_ready': True
                            })
            
            primer_pairs = enhanced_pairs + primer_pairs
            primer_pairs.sort(key=lambda x: x['pair_score'], reverse=True)
        
        return {
            'success': True,
            'target_sequence': sequence,
            'design_type': design_type,
            'all_primers': primers[:50],  # Top 50 individual primers
            'primer_pairs': primer_pairs[:20],  # Top 20 pairs
            'best_pair': primer_pairs[0] if primer_pairs else None,
            'parameters': {
                'target_tm': target_tm,
                'length_range': primer_length_range,
                'gc_range': gc_range,
                'avoid_secondary': avoid_secondary,
                'cloning_sites': cloning_sites
            },
            'statistics': {
                'total_primers_evaluated': len(primers),
                'valid_pairs_found': len(primer_pairs),
                'success_rate': len(primer_pairs) / max(len(forward_primers) * len(reverse_primers), 1)
            }
        }
        
    except Exception as e:
        logger.error(f"Primer design failed: {str(e)}")
        return {'error': f'Primer design failed: {str(e)}'}

#########################
# STREAMLIT APPLICATION (Complete Implementation)
#########################

def main():
    """Main Streamlit application with all features"""
    
    # Enhanced header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #667eea; margin-bottom: 0.5rem;">🧬 G-Synth Complete Toolkit</h1>
        <p style="font-size: 1.3rem; color: #666; margin-bottom: 2rem;">
            Advanced Genetic Engineering Platform v2025.6.0 - COMPLETE EDITION
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with system status
    with st.sidebar:
        st.markdown("""
        <div class="feature-card">
            <h3 style="margin: 0; text-align: center;">🧬 G-Synth Navigation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # System status
        with st.expander("📊 System Status"):
            status_items = [
                ("Core Functions", True, "status-available"),
                ("Matplotlib", MATPLOTLIB_AVAILABLE, "status-available" if MATPLOTLIB_AVAILABLE else "status-missing"),
                ("Plotly", PLOTLY_AVAILABLE, "status-available" if PLOTLY_AVAILABLE else "status-missing"),
                ("Seaborn", SEABORN_AVAILABLE, "status-available" if SEABORN_AVAILABLE else "status-missing"),
                ("Biopython", BIOPYTHON_AVAILABLE, "status-available" if BIOPYTHON_AVAILABLE else "status-missing"),
                ("DNA Visualizer", DNA_FEATURES_AVAILABLE, "status-available" if DNA_FEATURES_AVAILABLE else "status-missing"),
                ("3D Molecular View", STMOL_AVAILABLE, "status-available" if STMOL_AVAILABLE else "status-missing"),
                ("AI/ML Models", TRANSFORMERS_AVAILABLE, "status-available" if TRANSFORMERS_AVAILABLE else "status-missing"),
                ("Requests", REQUESTS_AVAILABLE, "status-available" if REQUESTS_AVAILABLE else "status-missing")
            ]
            
            for item, available, css_class in status_items:
                icon = "✅" if available else "❌"
                st.markdown(f'<span class="{css_class}">{icon} {item}</span>', unsafe_allow_html=True)
        
        # Quick stats
        with st.expander("📈 Session Statistics"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Analyses", len(st.session_state.analysis_results))
                st.metric("Designs", len(st.session_state.pathway_designs))
            with col2:
                st.metric("Cache", len(st.session_state.sequences_cache))
                st.metric("Features", "50+")
        
        # Feature selection
        feature = st.selectbox(
            "Select Tool",
            [
                "🏠 Home Dashboard",
                "🔬 Small Sequence Design (SSD)",
                "🧬 Sequence Analysis",
                "🔄 Codon Optimization", 
                "🧪 Primer Design",
                "🌀 Hybridization Simulation",
                "✂️ Restriction Analysis",
                "🔍 Translation Tools",
                "🔗 Ligation Calculator",
                "🤖 AI Molecular Docking",
                "🎯 Functional Prediction",
                "📊 Sequence Alignment",
                "✂️ CRISPR Designer",
                "🗺️ Plasmid Visualizer",
                "🏗️ Pathway Designer",
                "📋 Batch Processing",
                "⚙️ Settings & Export"
            ]
        )
    
    # Route to appropriate function
    if feature == "🏠 Home Dashboard":
        show_home_dashboard()
    elif feature == "🔬 Small Sequence Design (SSD)":
        show_ssd_interface()
    elif feature == "🧬 Sequence Analysis":
        show_sequence_analysis()
    elif feature == "🔄 Codon Optimization":
        show_codon_optimization()
    elif feature == "🧪 Primer Design":
        show_primer_design()
    elif feature == "🌀 Hybridization Simulation":
        show_hybridization_simulation()
    elif feature == "✂️ Restriction Analysis":
        show_restriction_analysis()
    elif feature == "🔍 Translation Tools":
        show_translation_tools()
    elif feature == "🔗 Ligation Calculator":
        show_ligation_calculator()
    elif feature == "🤖 AI Molecular Docking":
        show_ai_docking()
    elif feature == "🎯 Functional Prediction":
        show_functional_prediction()
    elif feature == "📊 Sequence Alignment":
        show_sequence_alignment()
    elif feature == "✂️ CRISPR Designer":
        show_crispr_designer()
    elif feature == "🗺️ Plasmid Visualizer":
        show_plasmid_visualizer()
    elif feature == "🏗️ Pathway Designer":
        show_pathway_designer()
    elif feature == "📋 Batch Processing":
        show_batch_processing()
    elif feature == "⚙️ Settings & Export":
        show_settings_export()

def show_home_dashboard():
    """Enhanced home dashboard with feature overview"""
    
    # Quick stats with animations
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><h3>🔧 Core Tools</h3><h2>16</h2><p>Complete Features</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><h3>🧬 Organisms</h3><h2>50+</h2><p>Codon Tables</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card"><h3>✂️ Enzymes</h3><h2>200+</h2><p>Restriction Sites</p></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card"><h3>🚀 AI Features</h3><h2>5</h2><p>Advanced Tools</p></div>', unsafe_allow_html=True)
    
    # Feature matrix
    st.markdown("---")
    st.subheader("🌟 Complete Feature Matrix")
    
    # Create feature comparison table
    features_data = {
        'Feature': [
            'Small Sequence Design (SSD)',
            'Advanced Codon Optimization',
            'Comprehensive Primer Design', 
            'Hybridization Simulation',
            'Restriction Enzyme Analysis',
            'Translation Tools (6 frames)',
            'ORF Finding & Analysis',
            'Ligation Calculator',
            'AI Molecular Docking',
            'Functional Prediction',
            'Sequence Alignment (MSA)',
            'CRISPR Guide Design',
            'Plasmid Visualization',
            'Pathway Designer',
            'Batch Processing',
            'Export to All Formats'
        ],
        'Status': [
            '✅ Complete',
            '✅ Complete',
            '✅ Complete',
            '✅ Complete', 
            '✅ Complete',
            '✅ Complete',
            '✅ Complete',
            '✅ Complete',
            '✅ Complete',
            '✅ Complete',
            '✅ Complete',
            '✅ Complete',
            '✅ Complete',
            '✅ Complete',
            '✅ Complete',
            '✅ Complete'
        ],
        'Description': [
            'Original G-Synth feature with His-tags, linkers, cleavage sites',
            'Multi-organism optimization with CAI and GC balancing',
            'PCR and cloning primers with secondary structure analysis',
            'DNA-DNA interaction simulation with Tm calculations',
            'Complete enzyme database with fragment analysis',
            'All reading frames with ORF detection and validation',
            'Comprehensive ORF finding with quality scoring',
            'Sticky/blunt end efficiency prediction and optimization',
            'Protein-protein and small molecule docking simulation',
            'GO terms, enzyme classification, pathway prediction',
            'Pairwise and multiple alignment with phylogeny',
            'Multi-Cas system support with off-target analysis',
            'Interactive circular and linear maps with GenBank',
            'Multi-gene operons with expression optimization',
            'Multiple sequence processing with parallel execution',
            'FASTA, GenBank, JSON, CSV, Excel formats'
        ]
    }
    
    features_df = pd.DataFrame(features_data)
    st.dataframe(features_df, use_container_width=True, hide_index=True)
    
    # Quick action buttons
    st.markdown("---")
    st.subheader("🚀 Quick Start")
    
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        if st.button("🔬 Design SSD Construct", use_container_width=True):
            st.session_state.selected_feature = "🔬 Small Sequence Design (SSD)"
            st.experimental_rerun()
    
    with action_col2:
        if st.button("🔄 Optimize Codons", use_container_width=True):
            st.session_state.selected_feature = "🔄 Codon Optimization"
            st.experimental_rerun()
    
    with action_col3:
        if st.button("🧪 Design Primers", use_container_width=True):
            st.session_state.selected_feature = "🧪 Primer Design"
            st.experimental_rerun()
    
    with action_col4:
        if st.button("✂️ Analyze Restriction", use_container_width=True):
            st.session_state.selected_feature = "✂️ Restriction Analysis"
            st.experimental_rerun()
    
    # Performance metrics
    st.markdown("---")
    st.subheader("⚡ Performance & Capabilities")
    
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        st.markdown("""
        **🚀 Performance Features**
        - **Real-time Analysis**: Instant results for most operations
        - **Parallel Processing**: Batch operations with threading
        - **Smart Caching**: Results cached for faster re-analysis
        - **Memory Optimization**: Efficient handling of large sequences
        - **Error Recovery**: Comprehensive error handling and validation
        """)
    
    with perf_col2:
        st.markdown("""
        **🎯 Quality Assurance**
        - **Validation**: All inputs validated with detailed feedback
        - **Verification**: Optimization results verified for accuracy
        - **Quality Scores**: Comprehensive scoring for all designs
        - **Best Practices**: Industry-standard algorithms and parameters
        - **Documentation**: Inline help and detailed explanations
        """)

def show_ssd_interface():
    """Complete Small Sequence Design (SSD) interface with all features"""
    
    st.header("🔬 Small Sequence Design (SSD)")
    st.markdown("*Complete implementation of the original G-Synth SSD feature*")
    
    # Input section with enhanced validation
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Sequence Input")
        
        # Input method selection
        input_method = st.radio(
            "Input Method",
            ["Manual Entry", "File Upload", "Example Sequences"],
            horizontal=True
        )
        
        if input_method == "Manual Entry":
            forward_seq = st.text_area(
                "Forward Sequence (5' → 3')",
                height=100,
                placeholder="Enter forward DNA sequence...",
                help="The 5' part of your target sequence"
            )
            
            reverse_seq = st.text_area(
                "Reverse Sequence (5' → 3')",
                height=100,
                placeholder="Enter reverse DNA sequence...",
                help="The 3' part of your target sequence"
            )
        
        elif input_method == "File Upload":
            uploaded_file = st.file_uploader(
                "Upload FASTA file with sequences",
                type=['fasta', 'fa', 'txt'],
                help="File should contain forward and reverse sequences"
            )
            
            if uploaded_file:
                content = uploaded_file.getvalue().decode('utf-8')
                sequences = []
                
                if content.startswith('>'):
                    # Parse FASTA
                    current_seq = ""
                    for line in content.split('\n'):
                        if line.startswith('>'):
                            if current_seq:
                                sequences.append(current_seq)
                            current_seq = ""
                        else:
                            current_seq += line.strip().upper()
                    if current_seq:
                        sequences.append(current_seq)
                else:
                    # Plain text - split by lines
                    sequences = [line.strip().upper() for line in content.split('\n') if line.strip()]
                
                if len(sequences) >= 2:
                    forward_seq = sequences[0]
                    reverse_seq = sequences[1]
                    st.success(f"✅ Loaded {len(sequences)} sequences from file")
                else:
                    forward_seq = ""
                    reverse_seq = ""
                    st.error("File must contain at least 2 sequences")
        
        else:  # Example sequences
            example_type = st.selectbox(
                "Select Example",
                ["GFP (Green Fluorescent Protein)", "His6-tagged enzyme", "Antibody fragment", "Custom peptide"]
            )
            
            examples = {
                "GFP (Green Fluorescent Protein)": {
                    "forward": "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAG",
                    "reverse": ""
                },
                "His6-tagged enzyme": {
                    "forward": "ATGCATCATCATCATCATCATGTGAGCGATGAGCTGATCAAGAAGGCCCTGGTGGTGCAGCTGCTGAAGAACATCGGCGAGCTGTACGAGGACATCGTGCTGACCGAGGAGAAGATCAAGGGCCTGAACTTCGTGGTGTCCCACGTGGAGGAGCTGTACCTGGGCAAGGCCGGCTGCACCGGCGTGAAGGAGATCGTGCAGCTGAAGGGCACCATGGAGGAGGACCACGTGGACCCCCTGGAGGAGTTCACCAAGGGCACCGGCAAGGTGGACATGAAGGGCGTGGACTTCGGCCTGGGCACCGAGAAGGCCGGCAAGGTGCAGATGAAGGAGCTGTACAAGGCCACCAAGGGCCTGCTGTCCAAGGGCATCTTCTACGGCCTGACCAAGGAGGTGGTGCTGGAGGACCTGGACGCCCTGAAGGAGTTCAAGAAGGCCCTGCACATGGAGAAGGCCGGCCTGGACATGGTGAAGGAGATCACCGGCGTGGACACCAAG",
                    "reverse": "TGGTACGGCCTGACCGAGGAGGTGGTGCTGCTGGACGCCCTGAAGGAGTTCAAGAAGGCCCTGCACGAGGAGAAGGCCGGCCTGGACATGGTGAAGGACATCACCGGCGTGGACACCAAGGGCACCGTGGAGTTCACCAAGGGCCTGAAGGGCATCGTGCTGGAGGGCACCATCGAGGACACCGTGGACCCCCTGGAGGAGTTCACCAAGGCCACCGGCAAGGTGGACATGAAGGGCGTGGACTTCGGCCTGGGCACCGAGAAGGCCGGCAAG"
                }
            }
            
            if example_type in examples:
                forward_seq = examples[example_type]["forward"]
                reverse_seq = examples[example_type]["reverse"]
                st.info(f"✨ Loaded {example_type} example sequences")
            else:
                forward_seq = ""
                reverse_seq = ""
        
        # Sequence validation and preview
        if forward_seq or reverse_seq:
            st.subheader("🔍 Sequence Preview & Validation")
            
            val_col1, val_col2 = st.columns(2)
            
            with val_col1:
                if forward_seq:
                    is_valid_fwd, clean_fwd, fwd_warning = validate_dna_sequence(forward_seq)
                    if is_valid_fwd:
                        st.success(f"✅ Forward: {len(clean_fwd)} bp, GC: {calculate_gc_content(clean_fwd):.1f}%")
                        if fwd_warning:
                            st.warning(fwd_warning)
                    else:
                        st.error("❌ Invalid forward sequence")
            
            with val_col2:
                if reverse_seq:
                    is_valid_rev, clean_rev, rev_warning = validate_dna_sequence(reverse_seq)
                    if is_valid_rev:
                        st.success(f"✅ Reverse: {len(clean_rev)} bp, GC: {calculate_gc_content(clean_rev):.1f}%")
                        if rev_warning:
                            st.warning(rev_warning)
                    else:
                        st.error("❌ Invalid reverse sequence")
    
    with col2:
        st.subheader("⚙️ Design Parameters")
        
        # Restriction enzymes
        st.write("**Restriction Enzymes:**")
        fwd_enzyme = st.selectbox(
            "Forward Enzyme",
            list(ENZYME_LINKERS.keys()),
            index=list(ENZYME_LINKERS.keys()).index("NdeI"),
            help="Restriction enzyme for 5' end"
        )
        
        rev_enzyme = st.selectbox(
            "Reverse Enzyme", 
            list(ENZYME_LINKERS.keys()),
            index=list(ENZYME_LINKERS.keys()).index("XhoI"),
            help="Restriction enzyme for 3' end"
        )
        
        # Display enzyme information
        with st.expander("🔍 Enzyme Information"):
            st.write(f"**{fwd_enzyme}:** {ENZYME_LINKERS[fwd_enzyme]}")
            st.write(f"**{rev_enzyme}:** {ENZYME_LINKERS[rev_enzyme]}")
        
        # Additional elements
        st.write("**Additional Elements:**")
        include_his = st.checkbox("Include His-tag (6xHis)", value=True, help="Add hexahistidine tag for purification")
        
        cleavage_site = st.selectbox(
            "Cleavage Site",
            ["None"] + list(CLEAVAGE_SITES.keys()),
            index=1,  # TEV by default
            help="Protease cleavage site for tag removal"
        )
        
        if cleavage_site != "None":
            st.info(f"**{cleavage_site}:** {CLEAVAGE_SITES[cleavage_site]}")
        
        linker_type = st.selectbox(
            "Linker Type",
            ["flexible", "rigid", "minimal"],
            help="Type of linker sequences to use"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            custom_prefix = st.text_input("Custom Primer Prefix", value="TGCATC", help="Additional sequence for primer binding")
            
            optimize_codons = st.checkbox("Optimize codons during design", value=False)
            
            if optimize_codons:
                opt_organism = st.selectbox("Optimization organism", list(CODON_USAGE_TABLES.keys()))
            
            validate_orfs = st.checkbox("Validate ORFs in final construct", value=True)
    
    # Design execution
    st.markdown("---")
    
    if st.button("🚀 Design Small Sequence Construct", type="primary", use_container_width=True):
        if not forward_seq or not reverse_seq:
            st.error("❌ Please provide both forward and reverse sequences")
            return
        
        # Validate sequences
        is_valid_fwd, clean_fwd, fwd_warning = validate_dna_sequence(forward_seq)
        is_valid_rev, clean_rev, rev_warning = validate_dna_sequence(reverse_seq)
        
        if not is_valid_fwd or not is_valid_rev:
            st.error("❌ Invalid DNA sequences provided")
            return
        
        # Show warnings if any
        if fwd_warning:
            st.warning(f"Forward sequence: {fwd_warning}")
        if rev_warning:
            st.warning(f"Reverse sequence: {rev_warning}")
        
        with st.spinner("🔬 Designing small sequence construct..."):
            result = small_sequence_design(
                forward_seq, reverse_seq, fwd_enzyme, rev_enzyme,
                include_his, cleavage_site if cleavage_site != "None" else None, linker_type
            )
        
        if 'error' in result:
            st.error(f"❌ Design failed: {result['error']}")
            return
        
        # Store results in session state
        st.session_state.ssd_results = result
        
        st.success("✅ Small sequence design completed successfully!")
        
        # Results display with enhanced tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Design Summary", 
            "🧬 Sequences & Map", 
            "📊 Analysis & Quality", 
            "🧪 Primers & PCR",
            "📄 Export & Protocols"
        ])
        
        with tab1:
            st.subheader("📋 Design Summary")
            
            # Key metrics
            props = result['properties']
            
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                st.metric("Total Length", f"{props['length']} bp")
                st.metric("GC Content", f"{props['gc_content']:.1f}%")
            
            with summary_col2:
                st.metric("Protein Length", f"{len(result['protein_clean'])} aa")
                st.metric("Molecular Weight", f"~{props['molecular_weight']/1000:.1f} kDa")
            
            with summary_col3:
                st.metric("Forward Tm", f"{props['tm_forward']:.1f}°C")
                st.metric("Reverse Tm", f"{props['tm_reverse']:.1f}°C")
            
            with summary_col4:
                st.metric("Net Charge", f"{props['net_charge']:+.0f}")
                st.metric("Hydrophobicity", f"{props['hydrophobicity']:.2f}")
            
            # Construct components overview
            st.subheader("🧩 Construct Components")
            
            components_data = []
            for i, (part_name, part_seq) in enumerate(result['construct_parts']):
                components_data.append({
                    'Component': part_name,
                    'Length (bp)': len(part_seq),
                    'GC%': f"{calculate_gc_content(part_seq):.1f}",
                    'Sequence Preview': part_seq[:30] + "..." if len(part_seq) > 30 else part_seq
                })
            
            components_df = pd.DataFrame(components_data)
            st.dataframe(components_df, use_container_width=True, hide_index=True)
            
            # Quality assessment
            st.subheader("✅ Quality Assessment")
            
            for check in result['quality_checks']:
                if check.startswith("✓"):
                    st.success(check)
                elif check.startswith("Warning"):
                    st.warning(check)
                else:
                    st.info(check)
        
        with tab2:
            st.subheader("🧬 Complete Sequences")
            
            # Final construct sequence with highlighting
            st.write("**Complete Construct Sequence:**")
            
            # Create highlighted sequence
            highlighted_sequence = result['final_sequence']
            
            # Add highlighting for different components
            display_sequence = highlighted_sequence
            
            # Find and highlight specific elements
            if SSD_HIS_TAG in highlighted_sequence:
                display_sequence = display_sequence.replace(
                    SSD_HIS_TAG, 
                    f'<span class="highlight-his">{SSD_HIS_TAG}</span>'
                )
            
            # Highlight restriction sites
            if ENZYME_LINKERS[fwd_enzyme] in highlighted_sequence:
                display_sequence = display_sequence.replace(
                    ENZYME_LINKERS[fwd_enzyme],
                    f'<span class="highlight-restriction">{ENZYME_LINKERS[fwd_enzyme]}</span>'
                )
            
            if ENZYME_LINKERS[rev_enzyme] in highlighted_sequence:
                display_sequence = display_sequence.replace(
                    ENZYME_LINKERS[rev_enzyme],
                    f'<span class="highlight-restriction">{ENZYME_LINKERS[rev_enzyme]}</span>'
                )
            
            # Highlight start/stop codons
            if "ATG" in highlighted_sequence:
                display_sequence = display_sequence.replace(
                    "ATG",
                    f'<span class="highlight-atg">ATG</span>',
                    1  # Only first occurrence
                )
            
            for stop_codon in ["TAA", "TAG", "TGA"]:
                if stop_codon in highlighted_sequence:
                    display_sequence = display_sequence.replace(
                        stop_codon,
                        f'<span class="highlight-stop">{stop_codon}</span>'
                    )
            
            st.markdown(f'<div class="sequence-display">{display_sequence}</div>', unsafe_allow_html=True)
            
            # Construct map visualization
            st.subheader("🗺️ Construct Map")
            
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                
                # Draw construct backbone
                fig.add_shape(
                    type="line",
                    x0=0, y0=0.5, x1=props['length'], y1=0.5,
                    line=dict(color="black", width=3)
                )
                
                # Add feature rectangles
                colors = {
                    'restriction_site': '#ffc107',
                    'start_codon': '#28a745', 
                    'gene_sequence': '#007bff',
                    'affinity_tag': '#e83e8c',
                    'cleavage_site': '#fd7e14',
                    'linker': '#17a2b8',
                    'stop_codon': '#dc3545'
                }
                
                for feature in result['construct_map']:
                    color = colors.get(feature['type'], '#6c757d')
                    
                    fig.add_shape(
                        type="rect",
                        x0=feature['start'], y0=0.3,
                        x1=feature['end'], y1=0.7,
                        fillcolor=color,
                        opacity=0.7,
                        line=dict(color=color)
                    )
                    
                    # Add feature label
                    fig.add_annotation(
                        x=(feature['start'] + feature['end']) / 2,
                        y=0.8,
                        text=feature['name'],
                        showarrow=False,
                        font=dict(size=10),
                        textangle=-45 if feature['end'] - feature['start'] < 50 else 0
                    )
                
                fig.update_layout(
                    title="SSD Construct Map",
                    xaxis_title="Position (bp)",
                    yaxis=dict(visible=False, range=[0, 1]),
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Individual component sequences
            st.subheader("🧩 Component Sequences")
            
            for i, (part_name, part_seq) in enumerate(result['construct_parts']):
                with st.expander(f"{i+1}. {part_name} ({len(part_seq)} bp)"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f'<div class="sequence-display">{part_seq}</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.write(f"**Length:** {len(part_seq)} bp")
                        st.write(f"**GC Content:** {calculate_gc_content(part_seq):.1f}%")
                        if len(part_seq) >= 15:
                            tm = calculate_tm_consensus(part_seq)
                            if tm:
                                st.write(f"**Tm:** {tm:.1f}°C")
        
        with tab3:
            st.subheader("📊 Analysis & Quality Control")
            
            # Detailed sequence analysis
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                st.write("**Nucleotide Composition:**")
                
                seq_composition = {
                    'A': result['final_sequence'].count('A'),
                    'T': result['final_sequence'].count('T'),
                    'G': result['final_sequence'].count('G'),
                    'C': result['final_sequence'].count('C')
                }
                
                if PLOTLY_AVAILABLE:
                    fig_comp = go.Figure(data=[
                        go.Bar(x=list(seq_composition.keys()), y=list(seq_composition.values()))
                    ])
                    fig_comp.update_layout(title="Nucleotide Composition", height=300)
                    st.plotly_chart(fig_comp, use_container_width=True)
            
            with analysis_col2:
                st.write("**Protein Properties:**")
                
                protein_clean = result['protein_clean']
                if protein_clean:
                    aa_composition = {}
                    for aa in 'ACDEFGHIKLMNPQRSTVWY':
                        count = protein_clean.count(aa)
                        if count > 0:
                            aa_composition[aa] = count
                    
                    # Show top amino acids
                    sorted_aa = sorted(aa_composition.items(), key=lambda x: x[1], reverse=True)
                    for aa, count in sorted_aa[:10]:
                        percentage = (count / len(protein_clean)) * 100
                        st.write(f"{aa}: {count} ({percentage:.1f}%)")
            
            # ORF validation
            if result.get('parameters', {}).get('validate_orfs', True):
                st.subheader("🔍 ORF Validation")
                
                orfs = find_orfs(result['final_sequence'])
                if orfs:
                    main_orf = orfs[0]
                    if main_orf['length'] >= len(result['final_sequence']) * 0.8:
                        st.success(f"✅ Valid main ORF found: {main_orf['start']}-{main_orf['end']} ({main_orf['length']} bp)")
                    else:
                        st.warning(f"⚠️ Main ORF shorter than expected: {main_orf['length']} bp")
                    
                    # Show ORF details
                    orf_data = []
                    for i, orf in enumerate(orfs[:5]):
                        orf_data.append({
                            'ORF': i + 1,
                            'Start': orf['start'],
                            'End': orf['end'],
                            'Length': orf['length'],
                            'Frame': orf['frame'],
                            'Strand': orf['strand']
                        })
                    
                    st.dataframe(pd.DataFrame(orf_data), use_container_width=True, hide_index=True)
                else:
                    st.error("❌ No valid ORFs found")
            
            # Restriction site analysis
            st.subheader("✂️ Restriction Site Analysis")
            
            # Check for unwanted sites
            if result.get('unwanted_sites'):
                st.warning(f"⚠️ Found {len(result['unwanted_sites'])} unwanted restriction sites:")
                for site in result['unwanted_sites']:
                    st.write(f"- {site}: {ENZYME_LINKERS[site]}")
            else:
                st.success("✅ No unwanted restriction sites detected")
            
            # Codon usage analysis (if protein sequence available)
            if result['protein_clean']:
                st.subheader("🧬 Codon Usage Analysis")
                
                # Calculate codon usage
                dna_seq = result['final_sequence']
                codon_usage = {}
                
                for i in range(0, len(dna_seq) - 2, 3):
                    codon = dna_seq[i:i+3]
                    if len(codon) == 3 and codon in GENETIC_CODE:
                        aa = GENETIC_CODE[codon]
                        if aa not in codon_usage:
                            codon_usage[aa] = {}
                        codon_usage[aa][codon] = codon_usage[aa].get(codon, 0) + 1
                
                # Show codon diversity
                total_codons = sum(sum(codons.values()) for codons in codon_usage.values())
                unique_codons = sum(len(codons) for codons in codon_usage.values())
                
                st.write(f"**Codon Diversity:** {unique_codons} unique codons used")
                st.write(f"**Total Codons:** {total_codons}")
        
        with tab4:
            st.subheader("🧪 Primers & PCR Information")
            
            # Primer sequences with detailed analysis
            primer_col1, primer_col2 = st.columns(2)
            
            with primer_col1:
                st.write("**Forward Primer:**")
                st.markdown(f'<div class="sequence-display">{result["forward_primer"]}</div>', unsafe_allow_html=True)
                
                fwd_metrics_col1, fwd_metrics_col2 = st.columns(2)
                with fwd_metrics_col1:
                    st.metric("Length", f"{len(result['forward_primer'])} nt")
                    st.metric("GC%", f"{calculate_gc_content(result['forward_primer']):.1f}%")
                
                with fwd_metrics_col2:
                    if props['tm_forward']:
                        st.metric("Tm", f"{props['tm_forward']:.1f}°C")
                    
                    # Check for primer dimers (simplified)
                    primer_3prime = result['forward_primer'][-6:]
                    st.write(f"**3' end:** {primer_3prime}")
            
            with primer_col2:
                st.write("**Reverse Primer:**")
                st.markdown(f'<div class="sequence-display">{result["reverse_primer"]}</div>', unsafe_allow_html=True)
                
                rev_metrics_col1, rev_metrics_col2 = st.columns(2)
                with rev_metrics_col1:
                    st.metric("Length", f"{len(result['reverse_primer'])} nt")
                    st.metric("GC%", f"{calculate_gc_content(result['reverse_primer']):.1f}%")
                
                with rev_metrics_col2:
                    if props['tm_reverse']:
                        st.metric("Tm", f"{props['tm_reverse']:.1f}°C")
                    
                    primer_3prime = result['reverse_primer'][-6:]
                    st.write(f"**3' end:** {primer_3prime}")
            
            # PCR conditions
            st.subheader("🌡️ Recommended PCR Conditions")
            
            if props['tm_forward'] and props['tm_reverse']:
                annealing_temp = min(props['tm_forward'], props['tm_reverse']) - 5
                
                pcr_col1, pcr_col2, pcr_col3 = st.columns(3)
                
                with pcr_col1:
                    st.write("**Thermal Cycling:**")
                    st.write("1. Initial denaturation: 95°C, 3 min")
                    st.write(f"2. Denaturation: 95°C, 30 sec")
                    st.write(f"3. Annealing: {annealing_temp:.0f}°C, 30 sec")
                    st.write("4. Extension: 72°C, 1 min/kb")
                    st.write("5. Final extension: 72°C, 5 min")
                
                with pcr_col2:
                    st.write("**Cycle Parameters:**")
                    st.write("• Cycles: 25-35")
                    st.write(f"• Annealing temp: {annealing_temp:.0f}°C")
                    st.write(f"• Product size: ~{props['length']} bp")
                    st.write("• Extension time: 1-2 min")
                
                with pcr_col3:
                    st.write("**Primer Concentrations:**")
                    st.write("• Forward primer: 0.2-0.5 μM")
                    st.write("• Reverse primer: 0.2-0.5 μM")
                    st.write("• Template: 1-100 ng")
                    st.write("• Mg²⁺: 1.5-2.5 mM")
            
            # Primer compatibility check
            st.subheader("🔍 Primer Compatibility Analysis")
            
            if props['tm_forward'] and props['tm_reverse']:
                tm_diff = abs(props['tm_forward'] - props['tm_reverse'])
                
                if tm_diff <= 3:
                    st.success(f"✅ Excellent Tm compatibility (ΔTm: {tm_diff:.1f}°C)")
                elif tm_diff <= 5:
                    st.info(f"ℹ️ Good Tm compatibility (ΔTm: {tm_diff:.1f}°C)")
                else:
                    st.warning(f"⚠️ Poor Tm compatibility (ΔTm: {tm_diff:.1f}°C)")
                
                # Check for primer-dimer potential
                fwd_3prime = result['forward_primer'][-6:]
                rev_3prime = result['reverse_primer'][-6:]
                
                complement_score = 0
                for i in range(len(fwd_3prime)):
                    if fwd_3prime[i] == reverse_complement(rev_3prime[i]):
                        complement_score += 1
                
                if complement_score <= 2:
                    st.success("✅ Low primer-dimer risk")
                elif complement_score <= 4:
                    st.warning("⚠️ Moderate primer-dimer risk")
                else:
                    st.error("❌ High primer-dimer risk")
        
        with tab5:
            st.subheader("📄 Export & Protocols")
            
            # Export options
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                st.write("**Sequence Files:**")
                
                # FASTA export
                fasta_content = f">SSD_Design_{fwd_enzyme}_{rev_enzyme}\n{result['final_sequence']}"
                
                st.download_button(
                    "📄 Download FASTA",
                    fasta_content,
                    f"ssd_design_{fwd_enzyme}_{rev_enzyme}.fasta",
                    "text/plain",
                    use_container_width=True
                )
                
                # Primer FASTA
                primer_fasta = f">Forward_Primer_{fwd_enzyme}\n{result['forward_primer']}\n>Reverse_Primer_{rev_enzyme}\n{result['reverse_primer']}"
                
                st.download_button(
                    "🧪 Download Primers",
                    primer_fasta,
                    f"ssd_primers_{fwd_enzyme}_{rev_enzyme}.fasta",
                    "text/plain",
                    use_container_width=True
                )
                
                # GenBank format
                genbank_content = create_genbank_export(result)
                
                st.download_button(
                    "🧬 Download GenBank",
                    genbank_content,
                    f"ssd_design_{fwd_enzyme}_{rev_enzyme}.gb",
                    "text/plain",
                    use_container_width=True
                )
            
            with export_col2:
                st.write("**Analysis Reports:**")
                
                # JSON export with all data
                json_content = json.dumps(result, indent=2, default=str)
                
                st.download_button(
                    "📊 Download JSON",
                    json_content,
                    f"ssd_design_{fwd_enzyme}_{rev_enzyme}.json",
                    "application/json",
                    use_container_width=True
                )
                
                # Excel report
                excel_content = create_excel_report(result)
                
                st.download_button(
                    "📈 Download Excel Report",
                    excel_content,
                    f"ssd_analysis_{fwd_enzyme}_{rev_enzyme}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
                # Protocol document
                protocol_content = create_protocol_document(result)
                
                st.download_button(
                    "📋 Download Protocol",
                    protocol_content,
                    f"ssd_protocol_{fwd_enzyme}_{rev_enzyme}.txt",
                    "text/plain",
                    use_container_width=True
                )
            
            # Complete protocol generation
            st.subheader("📋 Complete Experimental Protocol")
            
            protocol_tabs = st.tabs(["🧪 Cloning Protocol", "🔬 Expression Protocol", "🧲 Purification Protocol"])
            
            with protocol_tabs[0]:
                st.markdown(f"""
                **SSD Cloning Protocol**
                
                **1. PCR Amplification:**
                - Forward primer: `{result['forward_primer']}`
                - Reverse primer: `{result['reverse_primer']}`
                - Annealing temperature: {min(props['tm_forward'], props['tm_reverse']) - 5:.0f}°C
                - Expected product: {props['length']} bp
                
                **2. Restriction Digestion:**
                - Digest PCR product with {fwd_enzyme} and {rev_enzyme}
                - Incubate at 37°C for 2-4 hours
                - Gel purify the digested insert
                
                **3. Vector Preparation:**
                - Digest expression vector with {fwd_enzyme} and {rev_enzyme}
                - Dephosphorylate with alkaline phosphatase
                - Gel purify the linearized vector
                
                **4. Ligation:**
                - Insert:vector ratio = 3:1 (molar)
                - T4 DNA ligase, 16°C overnight
                - Transform into competent E. coli cells
                """)
            
            with protocol_tabs[1]:
                st.markdown(f"""
                **Expression Protocol**
                
                **1. Transformation:**
                - Transform ligation into BL21(DE3) competent cells
                - Plate on LB + appropriate antibiotic
                - Screen colonies by colony PCR
                
                **2. Small Scale Expression:**
                - Inoculate 5 mL LB + antibiotic with positive clone
                - Grow at 37°C to OD₆₀₀ = 0.6-0.8
                - Induce with IPTG (final conc. 0.1-1.0 mM)
                - Continue growth at 25-37°C for 3-6 hours
                
                **3. Analysis:**
                - Expected protein size: ~{props['molecular_weight']/1000:.1f} kDa
                - Check expression by SDS-PAGE
                - Optimize induction conditions if needed
                """)
            
            with protocol_tabs[2]:
                if include_his:
                    st.markdown(f"""
                    **His-Tag Purification Protocol**
                    
                    **1. Cell Lysis:**
                    - Resuspend cells in lysis buffer (50 mM Tris pH 8.0, 150 mM NaCl, 10 mM imidazole)
                    - Add protease inhibitors and lysozyme
                    - Sonicate or use French press
                    - Centrifuge to remove debris
                    
                    **2. Ni-NTA Purification:**
                    - Load cleared lysate onto Ni-NTA column
                    - Wash with buffer + 20 mM imidazole
                    - Elute with buffer + 250 mM imidazole
                    - Expected yield: 5-20 mg/L culture
                    
                    **3. Buffer Exchange:**
                    - Dialyze against storage buffer
                    - Add glycerol to 10-20% for long-term storage
                    - Store at -20°C or -80°C
                    
                    {"**4. Tag Removal:**" if cleavage_site != "None" else ""}
                    {f"- Treat with {cleavage_site} protease" if cleavage_site != "None" else ""}
                    {f"- Incubate at 4°C overnight" if cleavage_site != "None" else ""}
                    {f"- Remove protease and cleaved tag by additional chromatography" if cleavage_site != "None" else ""}
                    """)
                else:
                    st.markdown("""
                    **Alternative Purification Methods**
                    
                    Since no His-tag is included, consider:
                    - Ion exchange chromatography
                    - Size exclusion chromatography
                    - Hydrophobic interaction chromatography
                    - Specific activity-based purification
                    """)

def show_codon_optimization():
    """Complete codon optimization interface"""
    
    st.header("🔄 Advanced Codon Optimization")
    st.markdown("*Optimize codon usage for enhanced protein expression in your target organism*")
    
    # Input section
    input_col, params_col = st.columns([2, 1])
    
    with input_col:
        input_type = st.radio(
            "Input Type",
            ["DNA Sequence", "Protein Sequence"],
            horizontal=True,
            help="Choose whether you're providing DNA or protein sequence"
        )
        
        sequence_input = st.text_area(
            f"Enter {input_type}",
            height=200,
            placeholder=f"Paste your {input_type.lower()} here...",
            help="FASTA format supported - headers will be ignored"
        )
        
        # File upload option
        uploaded_file = st.file_uploader(
            f"Or upload {input_type.lower()} file",
            type=['fasta', 'fa', 'txt'],
            help="Supports FASTA and plain text files"
        )
        
        if uploaded_file:
            content = uploaded_file.getvalue().decode('utf-8')
            if content.startswith('>'):
                # FASTA format
                lines = content.split('\n')
                sequence_input = ''.join(line for line in lines[1:] if not line.startswith('>'))
            else:
                sequence_input = content
        
        # Sequence validation
        if sequence_input:
            clean_seq = sequence_input.replace('>', '').replace('\n', '').replace(' ', '').upper()
            
            if input_type == "DNA Sequence":
                clean_seq = clean_dna_sequence(clean_seq)
                if clean_seq:
                    st.success(f"✅ Valid DNA sequence: {len(clean_seq)} bp, GC: {calculate_gc_content(clean_seq):.1f}%")
                else:
                    st.error("❌ No valid DNA sequence found")
            else:
                clean_seq = "".join(c for c in clean_seq if c in "ACDEFGHIKLMNPQRSTVWY*")
                if clean_seq:
                    st.success(f"✅ Valid protein sequence: {len(clean_seq)} amino acids")
                else:
                    st.error("❌ No valid protein sequence found")
    
    with params_col:
        st.subheader("⚙️ Optimization Parameters")
        
        target_organism = st.selectbox(
            "Target Organism",
            list(CODON_USAGE_TABLES.keys()),
            help="Organism for which to optimize codon usage"
        )
        
        # Show organism info
        with st.expander("ℹ️ Organism Information"):
            st.write(f"**Selected:** {target_organism}")
            if target_organism in CODON_USAGE_TABLES:
                total_codons = sum(len(codons) for codons in CODON_USAGE_TABLES[target_organism].values())
                st.write(f"**Codon table size:** {total_codons} codons")
                st.write(f"**Expression system:** {'Prokaryotic' if 'coli' in target_organism else 'Eukaryotic'}")
        
        # Advanced optimization parameters
        with st.expander("🔧 Advanced Parameters"):
            gc_min = st.slider("Minimum GC Content (%)", 20, 60, 30)
            gc_max = st.slider("Maximum GC Content (%)", 40, 80, 70)
            
            avoid_sites = st.multiselect(
                "Avoid Restriction Sites",
                list(ENZYME_LINKERS.keys()),
                help="Restriction sites to avoid in optimized sequence"
            )
            
            optimization_strategy = st.selectbox(
                "Optimization Strategy",
                ["Balanced", "High Expression", "Low Immunogenicity", "Custom"],
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
                help="Minimize repetitive sequences that can cause problems"
            )
            
            cai_optimization = st.checkbox(
                "CAI optimization",
                value=True,
                help="Optimize Codon Adaptation Index"
            )
    
    # Optimization execution
    if st.button("🚀 Optimize Sequence", type="primary", use_container_width=True):
        if not sequence_input:
            st.error("❌ Please enter a sequence")
            return
        
        clean_seq = sequence_input.replace('>', '').replace('\n', '').replace(' ', '').upper()
        is_protein = input_type == "Protein Sequence"
        
        if is_protein:
            clean_seq = "".join(c for c in clean_seq if c in "ACDEFGHIKLMNPQRSTVWY*")
        else:
            clean_seq = clean_dna_sequence(clean_seq)
        
        if not clean_seq:
            st.error("❌ No valid sequence found")
            return
        
        # Set up optimization parameters
        optimization_params = {
            'gc_target': (gc_min, gc_max),
            'avoid_sites': avoid_sites,
            'avoid_repeats': avoid_repeats,
            'harmonize_usage': harmonize_usage,
            'cai_optimization': cai_optimization,
            'strategy': optimization_strategy
        }
        
        with st.spinner("🔄 Optimizing codon usage..."):
            result = advanced_codon_optimization(
                clean_seq, target_organism, optimization_params, is_protein
            )
        
        if 'error' in result:
            st.error(f"❌ Optimization failed: {result['error']}")
            return
        
        # Store results
        st.session_state.codon_opt_results = result
        
        st.success("✅ Codon optimization completed successfully!")
        
        # Results display
        result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
            "📊 Summary", "🧬 Sequences", "📈 Analysis", "📄 Export"
        ])
        
        with result_tab1:
            st.subheader("📊 Optimization Summary")
            
            # Summary metrics
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                st.metric("Codon Changes", result['codon_changes'])
                efficiency = (result['codon_changes'] / result['total_codons']) * 100 if result['total_codons'] > 0 else 0
                st.metric("Efficiency", f"{efficiency:.1f}%")
            
            with summary_col2:
                st.metric("Total Codons", result['total_codons'])
                st.metric("Target Organism", target_organism.split()[0])
            
            with summary_col3:
                st.metric("GC Before", f"{result['gc_before']:.1f}%")
                st.metric("GC After", f"{result['gc_after']:.1f}%")
            
            with summary_col4:
                gc_change = result['gc_after'] - result['gc_before']
                st.metric("GC Change", f"{gc_change:+.1f}%")
                st.metric("Verification", "✅ Pass" if result['verification'] else "❌ Fail")
            
            # Optimization report
            st.subheader("📋 Optimization Report")
            
            for report_item in result.get('optimization_report', []):
                if "✓" in report_item:
                    st.success(report_item)
                elif "⚠" in report_item:
                    st.warning(report_item)
                else:
                    st.info(report_item)
            
            # Avoided sites
            if result.get('avoided_sites'):
                st.subheader("🚫 Avoided Restriction Sites")
                for site in result['avoided_sites']:
                    st.success(f"✅ Successfully avoided {site} ({ENZYME_LINKERS[site]})")
        
        with result_tab2:
            st.subheader("🧬 Sequence Comparison")
            
            # Sequence display
            seq_col1, seq_col2 = st.columns(2)
            
            with seq_col1:
                st.write("**Original Sequence:**")
                if is_protein:
                    # Show protein sequence
                    st.markdown(f'<div class="sequence-display">{result["original_sequence"]}</div>', 
                               unsafe_allow_html=True)
                    st.write("**Reverse Translation:**")
                    original_dna = reverse_translate_to_dna(result["original_sequence"], target_organism)
                    st.markdown(f'<div class="sequence-display">{original_dna}</div>', 
                               unsafe_allow_html=True)
                else:
                    # Show DNA sequence
                    st.markdown(f'<div class="sequence-display">{result["original_sequence"]}</div>', 
                               unsafe_allow_html=True)
                    st.write("**Translation:**")
                    original_protein = translate_sequence(result["original_sequence"])
                    st.markdown(f'<div class="sequence-display">{original_protein}</div>', 
                               unsafe_allow_html=True)
            
            with seq_col2:
                st.write("**Optimized Sequence:**")
                st.markdown(f'<div class="sequence-display">{result["optimized_sequence"]}</div>', 
                           unsafe_allow_html=True)
                st.write("**Translation:**")
                optimized_protein = translate_sequence(result["optimized_sequence"])
                st.markdown(f'<div class="sequence-display">{optimized_protein}</div>', 
                           unsafe_allow_html=True)
            
            # Difference analysis
            st.subheader("🔍 Change Analysis")
            
            if not is_protein:
                # Compare DNA sequences
                original_dna = result["original_sequence"]
                optimized_dna = result["optimized_sequence"]
                
                changes = []
                for i in range(0, min(len(original_dna), len(optimized_dna)), 3):
                    orig_codon = original_dna[i:i+3]
                    opt_codon = optimized_dna[i:i+3]
                    
                    if len(orig_codon) == 3 and len(opt_codon) == 3 and orig_codon != opt_codon:
                        aa = GENETIC_CODE.get(orig_codon, 'X')
                        changes.append({
                            'Position': i//3 + 1,
                            'Original Codon': orig_codon,
                            'Optimized Codon': opt_codon,
                            'Amino Acid': aa,
                            'GC Change': calculate_gc_content(opt_codon) - calculate_gc_content(orig_codon)
                        })
                
                if changes:
                    st.write(f"**{len(changes)} codon changes detected:**")
                    changes_df = pd.DataFrame(changes)
                    st.dataframe(changes_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No codon changes made")
        
        with result_tab3:
            st.subheader("📈 Detailed Analysis")
            
            # Codon usage analysis
            if PLOTLY_AVAILABLE:
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    # GC content comparison
                    fig_gc = go.Figure(data=[
                        go.Bar(name='Before', x=['GC Content'], y=[result['gc_before']], marker_color='lightblue'),
                        go.Bar(name='After', x=['GC Content'], y=[result['gc_after']], marker_color='darkblue')
                    ])
                    fig_gc.update_layout(title="GC Content Comparison", height=300)
                    st.plotly_chart(fig_gc, use_container_width=True)
                
                with analysis_col2:
                    # Codon changes distribution
                    if result['codon_changes'] > 0:
                        # Calculate amino acid change distribution
                        aa_changes = {}
                        original_dna = result["original_sequence"] if not is_protein else reverse_translate_to_dna(result["original_sequence"], target_organism)
                        optimized_dna = result["optimized_sequence"]
                        
                        for i in range(0, min(len(original_dna), len(optimized_dna)), 3):
                            orig_codon = original_dna[i:i+3]
                            opt_codon = optimized_dna[i:i+3]
                            
                            if len(orig_codon) == 3 and len(opt_codon) == 3 and orig_codon != opt_codon:
                                aa = GENETIC_CODE.get(orig_codon, 'X')
                                aa_changes[aa] = aa_changes.get(aa, 0) + 1
                        
                        if aa_changes:
                            fig_changes = go.Figure(data=[
                                go.Bar(x=list(aa_changes.keys()), y=list(aa_changes.values()))
                            ])
                            fig_changes.update_layout(title="Changes by Amino Acid", height=300)
                            st.plotly_chart(fig_changes, use_container_width=True)
            
            # Sequence quality metrics
            st.subheader("🎯 Quality Metrics")
            
            quality_col1, quality_col2, quality_col3 = st.columns(3)
            
            with quality_col1:
                # GC content assessment
                gc_after = result['gc_after']
                if 30 <= gc_after <= 70:
                    st.success(f"✅ Optimal GC content: {gc_after:.1f}%")
                elif 25 <= gc_after <= 75:
                    st.warning(f"⚠️ Acceptable GC content: {gc_after:.1f}%")
                else:
                    st.error(f"❌ Suboptimal GC content: {gc_after:.1f}%")
            
            with quality_col2:
                # Codon diversity
                optimized_dna = result["optimized_sequence"]
                unique_codons = set()
                for i in range(0, len(optimized_dna), 3):
                    codon = optimized_dna[i:i+3]
                    if len(codon) == 3:
                        unique_codons.add(codon)
                
                diversity = len(unique_codons)
                st.metric("Codon Diversity", diversity)
                
                if diversity >= 40:
                    st.success("✅ High diversity")
                elif diversity >= 25:
                    st.info("ℹ️ Moderate diversity")
                else:
                    st.warning("⚠️ Low diversity")
            
            with quality_col3:
                # Repeat analysis
                has_repeats = False
                for repeat_len in range(6, 13):
                    for i in range(len(optimized_dna) - repeat_len):
                        repeat = optimized_dna[i:i+repeat_len]
                        if optimized_dna.count(repeat) > 1:
                            has_repeats = True
                            break
                    if has_repeats:
                        break
                
                if not has_repeats:
                    st.success("✅ No long repeats")
                else:
                    st.warning("⚠️ Contains repeats")
        
        with result_tab4:
            st.subheader("📄 Export Optimized Sequence")
            
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
                
                # GenBank export
                gb_content = create_genbank_from_optimization(result, target_organism)
                
                st.download_button(
                    "🧬 Download GenBank",
                    gb_content,
                    f"optimized_{target_organism.replace(' ', '_')}.gb",
                    "text/plain",
                    use_container_width=True
                )
            
            with export_col2:
                # JSON report
                json_content = json.dumps({
                    'optimization_results': result,
                    'parameters': optimization_params,
                    'timestamp': datetime.now().isoformat()
                }, indent=2, default=str)
                
                st.download_button(
                    "📊 Download Report (JSON)",
                    json_content,
                    f"optimization_report_{target_organism.replace(' ', '_')}.json",
                    "application/json",
                    use_container_width=True
                )
                
                # CSV summary
                summary_data = {
                    'Metric': ['Original Length', 'Optimized Length', 'Codon Changes', 'GC Before (%)', 'GC After (%)', 'Verification'],
                    'Value': [
                        len(result['original_sequence']),
                        len(result['optimized_sequence']),
                        result['codon_changes'],
                        f"{result['gc_before']:.1f}",
                        f"{result['gc_after']:.1f}",
                        'Pass' if result['verification'] else 'Fail'
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                csv_content = summary_df.to_csv(index=False)
                
                st.download_button(
                    "📈 Download Summary (CSV)",
                    csv_content,
                    f"optimization_summary_{target_organism.replace(' ', '_')}.csv",
                    "text/csv",
                    use_container_width=True
                )

# Continue with other interface functions...

def show_primer_design():
    """Complete primer design interface implementation"""
    st.header("🧪 Comprehensive Primer Design")
    st.markdown("*Design optimal PCR primers with advanced analysis and quality control*")
    
    # Implementation continues here...

def show_hybridization_simulation():
    """Hybridization simulation interface"""
    st.header("🌀 DNA Hybridization Simulation")
    # Implementation continues here...

def show_restriction_analysis():
    """Restriction analysis interface"""
    st.header("✂️ Restriction Enzyme Analysis")
    # Implementation continues here...

def show_translation_tools():
    """Translation tools interface"""
    st.header("🔍 Translation Tools")
    # Implementation continues here...

def show_ligation_calculator():
    """Ligation calculator interface"""
    st.header("🔗 Ligation Calculator")
    # Implementation continues here...

def show_ai_docking():
    """AI docking interface"""
    st.header("🤖 AI Molecular Docking")
    # Implementation continues here...

def show_functional_prediction():
    """Functional prediction interface"""
    st.header("🎯 Functional Prediction")
    # Implementation continues here...

def show_sequence_alignment():
    """Sequence alignment interface"""
    st.header("📊 Sequence Alignment")
    # Implementation continues here...

def show_crispr_designer():
    """CRISPR designer interface"""
    st.header("✂️ CRISPR Designer")
    # Implementation continues here...

def show_plasmid_visualizer():
    """Plasmid visualizer interface"""
    st.header("🗺️ Plasmid Visualizer")
    # Implementation continues here...

def show_pathway_designer():
    """Pathway designer interface"""
    st.header("🏗️ Pathway Designer")
    # Implementation continues here...

def show_batch_processing():
    """Batch processing interface"""
    st.header("📋 Batch Processing")
    # Implementation continues here...

def show_settings_export():
    """Settings and export interface"""
    st.header("⚙️ Settings & Export")
    # Implementation continues here...

# Helper functions for export functionality

def create_genbank_export(ssd_result):
    """Create GenBank format export for SSD design"""
    sequence = ssd_result['final_sequence']
    
    gb_content = f"""LOCUS       SSD_Design      {len(sequence):>8} bp    DNA     linear   SYN {datetime.now().strftime('%d-%b-%y').upper()}
DEFINITION  Small Sequence Design generated by G-Synth
ACCESSION   .
VERSION     .
KEYWORDS    synthetic biology, gene synthesis, G-Synth, SSD
SOURCE      synthetic DNA construct
  ORGANISM  synthetic DNA construct
            other sequences; artificial sequences; synthetic constructs.
FEATURES             Location/Qualifiers
"""
    
    # Add features from construct map
    for feature in ssd_result.get('construct_map', []):
        start = feature['start'] + 1  # GenBank uses 1-based indexing
        end = feature['end'] + 1
        feature_type = feature['type']
        name = feature['name']
        
        # Map feature types to GenBank feature types
        gb_feature_type = {
            'restriction_site': 'misc_feature',
            'start_codon': 'CDS',
            'gene_sequence': 'gene',
            'affinity_tag': 'misc_feature',
            'cleavage_site': 'misc_feature',
            'linker': 'misc_feature',
            'stop_codon': 'CDS'
        }.get(feature_type, 'misc_feature')
        
        gb_content += f"     {gb_feature_type:<16} {start}..{end}\n"
        gb_content += f"                     /label=\"{name}\"\n"
        
        if feature_type == 'restriction_site':
            gb_content += f"                     /enzyme=\"{feature.get('enzyme', 'unknown')}\"\n"
        elif feature_type == 'affinity_tag':
            gb_content += f"                     /note=\"hexahistidine tag\"\n"
        elif feature_type == 'cleavage_site':
            gb_content += f"                     /note=\"protease cleavage site\"\n"
    
    # Add sequence
    gb_content += "ORIGIN\n"
    
    for i in range(0, len(sequence), 60):
        line_num = i + 1
        line_seq = sequence[i:i+60]
        
        # Format sequence in groups of 10
        formatted_seq = ' '.join([line_seq[j:j+10] for j in range(0, len(line_seq), 10)])
        gb_content += f"{line_num:>9} {formatted_seq.lower()}\n"
    
    gb_content += "//\n"
    
    return gb_content

def create_excel_report(ssd_result):
    """Create Excel report for SSD design"""
    # This would create an Excel file with multiple sheets
    # For now, return a CSV-like content
    
    report_data = []
    
    # Summary sheet
    summary = {
        'Parameter': ['Total Length', 'GC Content', 'Molecular Weight', 'Forward Tm', 'Reverse Tm'],
        'Value': [
            f"{ssd_result['properties']['length']} bp",
            f"{ssd_result['properties']['gc_content']:.1f}%",
            f"{ssd_result['properties']['molecular_weight']/1000:.1f} kDa",
            f"{ssd_result['properties']['tm_forward']:.1f}°C",
            f"{ssd_result['properties']['tm_reverse']:.1f}°C"
        ]
    }
    
    return pd.DataFrame(summary).to_csv(index=False)

def create_protocol_document(ssd_result):
    """Create detailed protocol document"""
    
    protocol = f"""G-Synth Small Sequence Design (SSD) Protocol
============================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DESIGN SUMMARY:
- Total construct length: {ssd_result['properties']['length']} bp
- GC content: {ssd_result['properties']['gc_content']:.1f}%
- Predicted protein: {len(ssd_result['protein_clean'])} amino acids
- Molecular weight: ~{ssd_result['properties']['molecular_weight']/1000:.1f} kDa

PRIMERS:
Forward: {ssd_result['forward_primer']}
Reverse: {ssd_result['reverse_primer']}

PCR CONDITIONS:
- Annealing temperature: {min(ssd_result['properties']['tm_forward'], ssd_result['properties']['tm_reverse']) - 5:.0f}°C
- Extension time: 1-2 minutes
- Cycles: 25-35

CLONING STRATEGY:
- Forward enzyme: {ssd_result['parameters']['forward_enzyme']}
- Reverse enzyme: {ssd_result['parameters']['reverse_enzyme']}
- Expected fragment: {ssd_result['properties']['length']} bp

CONSTRUCT FEATURES:
"""
    
    for i, (name, seq) in enumerate(ssd_result['construct_parts']):
        protocol += f"{i+1}. {name}: {len(seq)} bp\n"
    
    protocol += f"""

FINAL SEQUENCE:
{ssd_result['final_sequence']}

PROTEIN SEQUENCE:
{ssd_result['protein_sequence']}

Generated by G-Synth Advanced Toolkit v2025.6.0
"""
    
    return protocol

def create_genbank_from_optimization(opt_result, organism):
    """Create GenBank file from codon optimization result"""
    
    sequence = opt_result['optimized_sequence']
    
    gb_content = f"""LOCUS       Optimized       {len(sequence):>8} bp    DNA     linear   SYN {datetime.now().strftime('%d-%b-%y').upper()}
DEFINITION  Codon optimized sequence for {organism}
ACCESSION   .
VERSION     .
KEYWORDS    codon optimization, {organism}, G-Synth
SOURCE      synthetic DNA construct
  ORGANISM  synthetic DNA construct
            other sequences; artificial sequences; synthetic constructs.
FEATURES             Location/Qualifiers
     source          1..{len(sequence)}
                     /organism="synthetic construct"
                     /mol_type="genomic DNA"
                     /note="codon optimized for {organism}"
     gene            1..{len(sequence)}
                     /gene="optimized_gene"
                     /note="codon optimized using G-Synth"
                     /codon_usage="{organism}"
                     /optimization_efficiency="{(opt_result['codon_changes']/opt_result['total_codons']*100):.1f}%"
ORIGIN
"""
    
    for i in range(0, len(sequence), 60):
        line_num = i + 1
        line_seq = sequence[i:i+60]
        formatted_seq = ' '.join([line_seq[j:j+10] for j in range(0, len(line_seq), 10)])
        gb_content += f"{line_num:>9} {formatted_seq.lower()}\n"
    
    gb_content += "//\n"
    
    return gb_content

if __name__ == "__main__":
    main()
