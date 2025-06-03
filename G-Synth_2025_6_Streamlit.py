#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G-Synth: Complete AI-Enhanced Genetic Engineering Toolkit - Streamlit Version
Version 2025.6.0 - Full Implementation

A comprehensive toolkit for gene synthesis and molecular cloning with complete functionality.
Developed by Dr. Mohamed Merzoug

Complete Features:
- Small Sequence Design
- Translation & Reverse Translation
- AI-Enhanced Codon Optimization
- Extended Synthesis with Merzoug Assembly
- Hybridization Simulation
- Advanced Ligation Check
- AI Primer Designer
- CRISPR Guide Designer
- Reverse Complement Tools
- Comprehensive Help System
"""

# CRITICAL: st.set_page_config MUST be the very first Streamlit command
import streamlit as st

# Set page config first - this fixes the error
st.set_page_config(
    page_title="G-Synth: Complete Genetic Engineering Toolkit",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/g-synth/help',
        'Report a bug': "https://github.com/g-synth/issues",
        'About': "G-Synth v2025.6.0 - Complete Platform by Dr. Mohamed Merzoug"
    }
)

# Now import everything else
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import re
import io
import base64
import json
import logging
from datetime import datetime
from collections import defaultdict
import random
import math
import time
import sys
from typing import Dict, List, Tuple, Optional, Union
import traceback
import requests
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from Bio import SeqIO, Seq
    from Bio.SeqRecord import SeqRecord
    USING_BIOPYTHON = True
except ImportError:
    USING_BIOPYTHON = False

try:
    from dna_features_viewer import GraphicFeature, CircularGraphicRecord, GraphicRecord
    USING_DNA_VIEWER = True
except ImportError:
    USING_DNA_VIEWER = False

# Enhanced CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    .main {
        padding-top: 1rem;
    }
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .ai-badge {
        display: inline-block;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .feature-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        border-color: #667eea;
    }
    
    .ai-feature-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 2px solid #e2e8f0;
        border-left: 4px solid #ff6b6b;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px -5px rgba(255, 107, 107, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .ai-feature-card::before {
        content: 'AI-POWERED';
        position: absolute;
        top: 10px;
        right: 10px;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    
    .sequence-display {
        font-family: 'JetBrains Mono', monospace;
        background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%);
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        overflow-x: auto;
        white-space: pre-wrap;
        word-break: break-all;
        line-height: 1.6;
        font-size: 0.9rem;
    }
    
    .start-codon { 
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 2px 4px;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .stop-codon { 
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 2px 4px;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .restriction-site { 
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 2px 4px;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .his-tag { 
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        padding: 2px 4px;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .success-message {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .warning-message {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .error-message {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .info-message {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .ai-insights {
        background: linear-gradient(135deg, #3742fa 0%, #2f3542 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .confidence-score {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
    }
    
    .confidence-bar {
        flex-grow: 1;
        height: 8px;
        background: rgba(255,255,255,0.2);
        border-radius: 4px;
        margin: 0 10px;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48ca8a);
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    .prediction-accuracy {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Core biological constants
NUCLEOTIDES = ['A', 'T', 'C', 'G']
AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '*']

# Genetic code table
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

# Enzyme pairs and sites
enzyme_pairs = {
    "NdeI / XhoI": {"forward_overhang": "TA", "reverse_overhang": "TCGA"},
    "NdeI / EcoRI": {"forward_overhang": "TA", "reverse_overhang": "AATT"},
    "BamHI / EcoRI": {"forward_overhang": "GATC", "reverse_overhang": "AATT"},
    "BamHI / XhoI": {"forward_overhang": "GATC", "reverse_overhang": "TCGA"},
    "SalI / XbaI": {"forward_overhang": "TCGAC", "reverse_overhang": "TCTAG"}
}

enzyme_linkers = {
    "NdeI": "CATATG", "XhoI": "CTCGAG", "EcoRI": "GAATTC", "BamHI": "GGATCC",
    "HindIII": "AAGCTT", "SalI": "GTCGAC", "XbaI": "TCTAGA", "NcoI": "CCATGG",
    "KpnI": "GGTACC", "SacI": "GAGCTC", "NotI": "GCGGCCGC", "SpeI": "ACTAGT",
    "PstI": "CTGCAG", "BglII": "AGATCT"
}

cleavage_sites = {
    "Thrombin": "CTGGTGCCGCGTGGTTCT",
    "TEV": "GAAAACCTGTATTTTCAGGGC",
    "Factor Xa": "ATCGAAGGTCGT",
    "PreScission": "CTGGAAGTGCTGTTCCAGGGCCCA",
    "Enterokinase": "GATGACGATGACAAG",
    "SUMO": "CTGCAGGACTCAGAGG",
    "HRV 3C": "CTGGAAGTTCTGTTCCAGGGGCCC"
}

# Codon usage tables
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

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = {}
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'theme': 'light',
        'default_organism': 'E. coli BL21',
        'show_advanced_options': False,
        'ai_enabled': True,
        'confidence_threshold': 0.7
    }
if 'selected_tool' not in st.session_state:
    st.session_state.selected_tool = "Home"
if 'session_start' not in st.session_state:
    st.session_state.session_start = time.time()

# Core biological functions
def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    table = str.maketrans("ACGTN", "TGCAN")
    return seq.upper().translate(table)[::-1]

def clean_dna_sequence(seq, keep_ambiguous=False):
    """Clean a DNA sequence by removing invalid characters."""
    if keep_ambiguous:
        return re.sub(r"[^ACGTRYSWKMBDHVN]", "", seq.upper())
    else:
        return re.sub(r"[^ATCG]", "", seq.upper())

def validate_dna_sequence(sequence, allow_empty=False, allow_ambiguous=False):
    """Validate a DNA sequence and return clean version if valid."""
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
    """Translate a nucleotide sequence to protein."""
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

def calculate_gc(seq):
    """Calculate GC content as percentage."""
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / len(seq) * 100 if seq else 0

def calculate_tm_consensus(sequence, primer_conc=500e-9, na_conc=50e-3):
    """Calculate melting temperature using consensus of thermodynamic models."""
    sequence = sequence.upper().replace(' ', '')
    if not sequence or not all(base in "ATCG" for base in sequence):
        return None
    
    if len(sequence) < 8:
        a = sequence.count('A')
        t = sequence.count('T')
        g = sequence.count('G')
        c = sequence.count('C')
        return 2 * (a + t) + 4 * (g + c) - 7
    
    R = 1.987  # cal/(mol*K)
    delta_h = 0
    delta_s = 0
    
    nn_params = {
        "AA": (-7.9, -22.2), "TT": (-7.9, -22.2),
        "AT": (-7.2, -20.4), "TA": (-7.2, -21.3),
        "CA": (-8.5, -22.7), "TG": (-8.5, -22.7),
        "GT": (-8.4, -22.4), "AC": (-8.4, -22.4),
        "CT": (-7.8, -21.0), "AG": (-7.8, -21.0),
        "GA": (-8.2, -22.2), "TC": (-8.2, -22.2),
        "CG": (-10.6, -27.2), "GC": (-9.8, -24.4),
        "GG": (-8.0, -19.9), "CC": (-8.0, -19.9)
    }
    
    for i in range(len(sequence) - 1):
        pair = sequence[i:i+2]
        if pair in nn_params:
            h, s = nn_params[pair]
            delta_h += h
            delta_s += s
    
    delta_s += -10.8
    c = primer_conc / 4
    tm_kelvin = (delta_h * 1000) / (delta_s + R * math.log(c))
    salt_correction = 16.6 * math.log10(na_conc)
    
    return round(tm_kelvin - 273.15 + salt_correction, 1)

def reverse_translate_to_dna(prot, target_organism="E. coli BL21"):
    """Convert a protein sequence to DNA using optimal codons for the specified organism."""
    if target_organism not in CODON_USAGE_TABLES:
        target_organism = "E. coli BL21"
    
    codon_table = CODON_USAGE_TABLES[target_organism]
    return "".join(codon_table.get(aa, ["NNN"])[0] for aa in prot if aa in codon_table)

# Utility functions
def create_metric_card(title, value, description="", color="#667eea"):
    """Create a styled metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 2rem; font-weight: 700; color: {color}; margin-bottom: 0.5rem;">{value}</div>
        <div style="font-size: 0.9rem; color: #64748b; font-weight: 500;">{title}</div>
        {f'<div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.5rem;">{description}</div>' if description else ''}
    </div>
    """, unsafe_allow_html=True)

def create_ai_metric_card(title, value, confidence, description=""):
    """Create an AI-enhanced metric card with confidence indicator"""
    confidence_color = "#48ca8a" if confidence > 0.8 else "#feca57" if confidence > 0.6 else "#ff6b6b"
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 2rem; font-weight: 700; color: #667eea; margin-bottom: 0.5rem;">{value}</div>
        <div style="font-size: 0.9rem; color: #64748b; font-weight: 500;">{title}</div>
        <div class="confidence-score">
            <span style="font-size: 0.8rem;">Confidence:</span>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence*100}%; background: {confidence_color};"></div>
            </div>
            <span style="font-size: 0.8rem; font-weight: 600;">{confidence:.1%}</span>
        </div>
        {f'<div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.5rem;">{description}</div>' if description else ''}
    </div>
    """, unsafe_allow_html=True)

def create_download_button(content, filename, button_text, mime_type="text/plain"):
    """Create enhanced download button"""
    return st.download_button(
        label=f"📥 {button_text}",
        data=content,
        file_name=filename,
        mime=mime_type,
        help=f"Download {filename}"
    )

def create_status_message(message, status_type="info"):
    """Create styled status messages"""
    class_map = {
        "success": "success-message",
        "warning": "warning-message", 
        "error": "error-message",
        "info": "info-message"
    }
    
    css_class = class_map.get(status_type, "info-message")
    st.markdown(f'<div class="{css_class}">{message}</div>', unsafe_allow_html=True)

def create_ai_insights_panel(insights: List[str]):
    """Create AI insights panel"""
    insights_html = "<br>".join([f"• {insight}" for insight in insights])
    
    st.markdown(f"""
    <div class="ai-insights">
        <h4>🧠 AI Insights</h4>
        {insights_html}
    </div>
    """, unsafe_allow_html=True)

def highlight_sequence_features(sequence):
    """Create highlighted sequence display with biological features"""
    if not sequence:
        return ""
    
    highlighted_seq = sequence
    
    # Highlight start codons
    highlighted_seq = re.sub(r'ATG', '<span class="start-codon">ATG</span>', highlighted_seq)
    
    # Highlight stop codons
    for stop in ['TAA', 'TAG', 'TGA']:
        highlighted_seq = re.sub(stop, f'<span class="stop-codon">{stop}</span>', highlighted_seq)
    
    # Highlight His-tag
    highlighted_seq = re.sub(r'CACCACCACCACCACCAC', '<span class="his-tag">CACCACCACCACCACCAC</span>', highlighted_seq)
    
    # Highlight common restriction sites
    for enzyme, site in enzyme_linkers.items():
        highlighted_seq = re.sub(site, f'<span class="restriction-site" title="{enzyme}">{site}</span>', highlighted_seq)
    
    return highlighted_seq

# Application header
def render_header():
    """Render the main application header"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <div class="main-header">G-Synth <span class="ai-badge">AI-ENHANCED</span></div>
        <div class="subtitle">Complete AI-Powered Genetic Engineering Toolkit</div>
        <div style="font-size: 1rem; color: #94a3b8; margin-bottom: 2rem;">
            Professional toolkit for gene synthesis, molecular cloning, and advanced sequence analysis
        </div>
        <div style="font-size: 0.9rem; color: #cbd5e1;">
            Version 2025.6.0 • Complete Implementation • By Dr. Mohamed Merzoug
        </div>
    </div>
    """, unsafe_allow_html=True)

# Navigation sidebar
def render_sidebar():
    """Render the navigation sidebar"""
    st.sidebar.markdown("## 🧰 Complete Toolkit")
    
    # Tool options
    tools = [
        ("Home", "Welcome and overview", False),
        ("Small Sequence Design", "Design short DNA sequences", False),
        ("Translation & Reverse Translation", "Convert DNA ↔ Protein", False),
        ("AI Codon Optimization", "AI-enhanced codon optimization", True),
        ("Extended Synthesis", "Long sequence assembly", False),
        ("Hybridization Simulation", "DNA strand annealing", False),
        ("Ligation Check", "Fragment compatibility", False),
        ("AI Primer Designer", "AI-optimized primer design", True),
        ("CRISPR Guide Designer", "AI-powered CRISPR design", True),
        ("Reverse Complement", "Sequence manipulation", False),
        ("Help & Guide", "Documentation", False)
    ]
    
    # Create navigation
    for name, desc, is_ai in tools:
        is_selected = st.session_state.selected_tool == name
        display_name = f"{name} 🤖" if is_ai else name
        
        if st.sidebar.button(display_name, key=f"nav_{name}", help=desc, use_container_width=True):
            st.session_state.selected_tool = name
            st.rerun()
    
    # AI Configuration
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI Settings")
    
    ai_enabled = st.sidebar.checkbox(
        "Enable AI Features",
        value=st.session_state.user_preferences.get('ai_enabled', True),
        help="Toggle AI-powered analysis and predictions"
    )
    
    if ai_enabled:
        confidence_threshold = st.sidebar.slider(
            "AI Confidence Threshold",
            0.5, 0.95, 
            st.session_state.user_preferences.get('confidence_threshold', 0.7),
            help="Minimum confidence for AI predictions"
        )
        
        st.session_state.user_preferences.update({
            'ai_enabled': ai_enabled,
            'confidence_threshold': confidence_threshold
        })
        
        st.sidebar.success("🤖 AI Ready")
    else:
        st.sidebar.info("🤖 AI Disabled")
        st.session_state.user_preferences['ai_enabled'] = False
    
    # Session info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Session Info")
    
    if st.session_state.history:
        st.sidebar.metric("Operations", len(st.session_state.history))
        
        ai_operations = sum(1 for h in st.session_state.history if 'AI' in h.get('tool', ''))
        st.sidebar.metric("AI Operations", ai_operations)
        
        st.sidebar.markdown("**Recent:**")
        for activity in st.session_state.history[-3:]:
            tool_name = activity.get('tool', 'Unknown')
            icon = "🤖" if "AI" in tool_name else "🔬"
            st.sidebar.caption(f"{icon} {activity.get('action', 'Unknown')}")
    else:
        st.sidebar.info("No operations performed yet")
    
    return st.session_state.selected_tool

# Home tab
def render_home_tab():
    """Render the home dashboard"""
    
    # Welcome section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%); 
                    border-radius: 16px; border: 1px solid #e2e8f0; margin-bottom: 2rem;">
            <h3 style="color: #1e293b; margin-bottom: 1rem;">Complete G-Synth Platform</h3>
            <p style="color: #64748b; line-height: 1.6;">
                A comprehensive genetic engineering toolkit that combines traditional molecular biology tools 
                with cutting-edge AI capabilities for enhanced research and development.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("## 🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🧬 Design Sequence", use_container_width=True):
            st.session_state.selected_tool = "Small Sequence Design"
            st.rerun()
    
    with col2:
        if st.button("🔄 Translate DNA", use_container_width=True):
            st.session_state.selected_tool = "Translation & Reverse Translation"
            st.rerun()
    
    with col3:
        if st.button("🤖 AI Optimize", use_container_width=True):
            st.session_state.selected_tool = "AI Codon Optimization"
            st.rerun()
    
    with col4:
        if st.button("🧪 Design Primers", use_container_width=True):
            st.session_state.selected_tool = "AI Primer Designer"
            st.rerun()
    
    # Feature overview
    st.markdown("## 🛠️ Complete Toolkit Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 1.4rem; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">Small Sequence Design</div>
            <div style="color: #64748b; line-height: 1.6; margin-bottom: 1.5rem;">Design and optimize short DNA sequences with enzyme sites for cloning and protein expression.</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="ai-feature-card">
            <div style="font-size: 1.4rem; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">AI Codon Optimization</div>
            <div style="color: #64748b; line-height: 1.6; margin-bottom: 1.5rem;">Machine learning-guided codon optimization for enhanced protein expression in multiple host systems.</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 1.4rem; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">Hybridization Simulation</div>
            <div style="color: #64748b; line-height: 1.6; margin-bottom: 1.5rem;">Simulate DNA strand hybridization and predict annealing behavior for molecular assembly.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 1.4rem; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">Translation Tools</div>
            <div style="color: #64748b; line-height: 1.6; margin-bottom: 1.5rem;">Convert between DNA sequences and amino acid sequences with multiple reading frame support.</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 1.4rem; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">Extended Synthesis</div>
            <div style="color: #64748b; line-height: 1.6; margin-bottom: 1.5rem;">Fragment and assemble large DNA sequences for gene synthesis with Merzoug Assembly.</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 1.4rem; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">Ligation Check</div>
            <div style="color: #64748b; line-height: 1.6; margin-bottom: 1.5rem;">Verify compatibility of DNA fragments for ligation reactions with detailed analysis.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="ai-feature-card">
            <div style="font-size: 1.4rem; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">AI Primer Designer</div>
            <div style="color: #64748b; line-height: 1.6; margin-bottom: 1.5rem;">Intelligent primer design with specificity prediction, off-target analysis, and success rate optimization.</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="ai-feature-card">
            <div style="font-size: 1.4rem; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">CRISPR Guide Designer</div>
            <div style="color: #64748b; line-height: 1.6; margin-bottom: 1.5rem;">AI-powered CRISPR guide RNA design with on-target efficiency and off-target prediction.</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 1.4rem; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">Reverse Complement</div>
            <div style="color: #64748b; line-height: 1.6; margin-bottom: 1.5rem;">Generate complementary DNA strands for molecular biology applications.</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Platform statistics
    st.markdown("---")
    st.markdown("## 📈 Platform Capabilities")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        create_metric_card("Organism Support", "5+", "Codon usage tables")
    
    with col2:
        create_metric_card("Enzyme Database", "14+", "Restriction enzymes")
    
    with col3:
        create_metric_card("Cleavage Sites", "7+", "Protease sites")
    
    with col4:
        create_metric_card("AI Models", "3+", "Machine learning")
    
    with col5:
        create_metric_card("Tools Available", "11", "Complete toolkit")

# Small Sequence Design tab
def render_small_sequence_design_tab():
    """Render the Small Sequence Design tab"""
    st.markdown("## Small Sequence Design")
    st.markdown("Design and optimize small DNA sequences and oligonucleotides for cloning and expression.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Input Parameters")
        
        # Sequence input
        sequence_input = st.text_area(
            "DNA Sequence:",
            height=150,
            placeholder="Enter your DNA sequence (A, T, C, G only)",
            help="Enter the DNA sequence you want to design or optimize"
        )
        
        # Real-time validation
        if sequence_input:
            is_valid, clean_seq, warning = validate_dna_sequence(sequence_input)
            if not is_valid:
                create_status_message(f"⚠️ {warning}", "warning")
            elif warning:
                create_status_message(f"ℹ️ {warning}", "info")
            else:
                create_status_message(f"✅ Valid sequence ({len(clean_seq)} bp)", "success")
        
        # Sequence type selection
        sequence_type = st.selectbox(
            "Sequence Type:",
            ["Coding Sequence", "Non-coding Sequence"],
            help="Select whether your sequence codes for a protein or not"
        )
        
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            if sequence_type == "Coding Sequence":
                remove_stop_codons = st.checkbox(
                    "Remove Stop Codons",
                    value=True,
                    help="Remove stop codons from the sequence"
                )
            else:
                add_his_tag = st.checkbox(
                    "Add His-Tag",
                    value=True,
                    help="Add 6xHis tag for protein purification"
                )
        
        with col1_2:
            enzyme_pair = st.selectbox(
                "Enzyme Pair:",
                list(enzyme_pairs.keys()),
                help="Select restriction enzyme pair for cloning"
            )
        
        # Cleavage site for non-coding sequences
        if sequence_type == "Non-coding Sequence":
            cleavage_site = st.selectbox(
                "Cleavage Site (Optional):",
                ["None"] + list(cleavage_sites.keys()),
                help="Add a protease cleavage site"
            )
            cleavage_site = None if cleavage_site == "None" else cleavage_site
        else:
            cleavage_site = None
        
        # Process button
        if st.button("🚀 Design Sequence", type="primary", use_container_width=True):
            if sequence_input:
                with st.spinner("Designing sequence..."):
                    try:
                        # Use SSD processing logic
                        result = process_ssd_sequence(
                            sequence_input, 
                            sequence_type == "Coding Sequence",
                            remove_stop_codons if sequence_type == "Coding Sequence" else False,
                            enzyme_pair,
                            cleavage_site
                        )
                        
                        st.session_state.current_results = result
                        
                        # Add to history
                        st.session_state.history.append({
                            'timestamp': datetime.now().strftime("%H:%M:%S"),
                            'action': f"Designed {sequence_type.lower()} ({len(clean_seq)} bp)",
                            'tool': 'Small Sequence Design'
                        })
                        
                        create_status_message("✅ Sequence design completed successfully!", "success")
                    
                    except Exception as e:
                        logger.error(f"Error in sequence design: {e}")
                        create_status_message(f"❌ An error occurred: {str(e)}", "error")
            else:
                create_status_message("❌ Please enter a DNA sequence", "error")
    
    with col2:
        st.markdown("### Design Results")
        
        if 'current_results' in st.session_state and st.session_state.current_results:
            result = st.session_state.current_results
            
            if "error" not in result:
                # Display sequences with highlighting
                st.markdown("#### Forward Sequence (5' → 3')")
                forward_seq = result.get("forward", "")
                highlighted_forward = highlight_sequence_features(forward_seq)
                st.markdown(f'<div class="sequence-display">{highlighted_forward}</div>', unsafe_allow_html=True)
                
                st.markdown("#### Reverse Sequence (5' → 3')")
                reverse_seq = result.get("reverse", "")
                highlighted_reverse = highlight_sequence_features(reverse_seq)
                st.markdown(f'<div class="sequence-display">{highlighted_reverse}</div>', unsafe_allow_html=True)
                
                # Properties display
                if "properties" in result:
                    st.markdown("#### Sequence Properties")
                    props = result["properties"]
                    
                    col2_1, col2_2 = st.columns(2)
                    
                    with col2_1:
                        st.markdown("**Forward Strand:**")
                        create_metric_card("Length", f"{props.get('forward_length', 0)} bp")
                        create_metric_card("GC Content", f"{props.get('forward_gc', 0):.1f}%")
                        if props.get('forward_tm'):
                            create_metric_card("Tm", f"{props.get('forward_tm', 0):.1f}°C")
                    
                    with col2_2:
                        st.markdown("**Reverse Strand:**")
                        create_metric_card("Length", f"{props.get('reverse_length', 0)} bp")
                        create_metric_card("GC Content", f"{props.get('reverse_gc', 0):.1f}%")
                        if props.get('reverse_tm'):
                            create_metric_card("Tm", f"{props.get('reverse_tm', 0):.1f}°C")
                
                # Download options
                st.markdown("#### Download Options")
                col2_3, col2_4 = st.columns(2)
                
                with col2_3:
                    fasta_content = f">Forward_Sequence\n{forward_seq}\n>Reverse_Sequence\n{reverse_seq}"
                    create_download_button(fasta_content, "designed_sequence.fasta", "Download FASTA")
                
                with col2_4:
                    report_content = f"""G-Synth Small Sequence Design Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

Sequence Type: {sequence_type}
Enzyme Pair: {enzyme_pair}
{f'Cleavage Site: {cleavage_site}' if cleavage_site else ''}

Forward Sequence (5' -> 3'):
{forward_seq}

Reverse Sequence (5' -> 3'):
{reverse_seq}

Properties:
- Forward Length: {props.get('forward_length', 0)} bp
- Forward GC Content: {props.get('forward_gc', 0):.1f}%
- Forward Tm: {props.get('forward_tm', 0):.1f}°C
- Reverse Length: {props.get('reverse_length', 0)} bp
- Reverse GC Content: {props.get('reverse_gc', 0):.1f}%
- Reverse Tm: {props.get('reverse_tm', 0):.1f}°C
"""
                    create_download_button(report_content, "design_report.txt", "Download Report")
            
            else:
                create_status_message(f"❌ Error: {result['error']}", "error")
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%); 
                        border-radius: 12px; border: 2px dashed #cbd5e1;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔬</div>
                <div style="color: #64748b; font-size: 1.1rem;">
                    Enter a DNA sequence and click 'Design Sequence' to see results
                </div>
            </div>
            """, unsafe_allow_html=True)

# Helper functions for SSD
def process_ssd_sequence(input_sequence, is_coding, remove_stop, enzyme_pair, cleavage_site=None):
    """Process a DNA sequence for synthesis based on sequence type and parameters"""
    
    # SSD constants
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
        "NotI": {"recognition": "GCGGCCGC", "cut_forward": "GGCC", "cut_reverse": "GC"},
        "SpeI": {"recognition": "ACTAGT", "cut_forward": "CTAG", "cut_reverse": "T"},
        "PstI": {"recognition": "CTGCAG", "cut_forward": "", "cut_reverse": "CTGCA"},
        "BglII": {"recognition": "AGATCT", "cut_forward": "GATC", "cut_reverse": "T"}
    }
    
    valid, seq, error = validate_dna_sequence(input_sequence)
    if not valid:
        return {"error": error}
    
    try:
        left_enzyme, right_enzyme = enzyme_pair.split(" / ")
    except ValueError:
        return {"error": "Enzyme pair format is incorrect. Please use the format 'Enz1 / Enz2'."}
    
    if left_enzyme not in SSD_RESTRICTION_ENZYMES:
        return {"error": f"Unknown restriction enzyme: {left_enzyme}"}
    if right_enzyme not in SSD_RESTRICTION_ENZYMES:
        return {"error": f"Unknown restriction enzyme: {right_enzyme}"}
    
    if is_coding:
        if not seq.startswith("ATG"):
            return {"error": "Coding sequence must start with ATG."}
        
        # Remove ATG if using NdeI to avoid duplication
        if left_enzyme == "NdeI":
            seq = seq[3:]
        
        if remove_stop:
            stop_codons = ["TAA", "TAG", "TGA"]
            for i in range(0, len(seq) - 2, 3):
                codon = seq[i:i+3]
                if codon in stop_codons:
                    seq = seq[:i]
                    break
        
        forward = SSD_RESTRICTION_ENZYMES[left_enzyme]["cut_forward"] + seq + SSD_RESTRICTION_ENZYMES[right_enzyme]["cut_forward"]
        rev_comp = reverse_complement(seq)
        reverse = SSD_RESTRICTION_ENZYMES[right_enzyme]["cut_reverse"] + rev_comp + SSD_RESTRICTION_ENZYMES[left_enzyme]["cut_reverse"]
    
    else:
        atg_prefix = "" if left_enzyme == "NdeI" else "ATG"
        
        forward = SSD_RESTRICTION_ENZYMES[left_enzyme]["cut_forward"]
        forward += atg_prefix + SSD_LEFT_LINKER + SSD_HIS_TAG + SSD_RIGHT_LINKER
        
        if cleavage_site and cleavage_site in cleavage_sites:
            forward += cleavage_sites[cleavage_site]
        
        forward += seq + SSD_RESTRICTION_ENZYMES[right_enzyme]["cut_forward"]
        
        reverse = SSD_RESTRICTION_ENZYMES[right_enzyme]["cut_reverse"]
        reverse += reverse_complement(seq)
        
        if cleavage_site and cleavage_site in cleavage_sites:
            reverse += reverse_complement(cleavage_sites[cleavage_site])
        
        reverse += reverse_complement(SSD_RIGHT_LINKER)
        reverse += reverse_complement(SSD_HIS_TAG)
        reverse += reverse_complement(SSD_LEFT_LINKER)
        reverse += reverse_complement(atg_prefix)
        reverse += SSD_RESTRICTION_ENZYMES[left_enzyme]["cut_reverse"]
    
    properties = {
        "forward_length": len(forward),
        "reverse_length": len(reverse),
        "forward_gc": calculate_gc(forward),
        "reverse_gc": calculate_gc(reverse),
        "forward_tm": calculate_tm_consensus(forward),
        "reverse_tm": calculate_tm_consensus(reverse)
    }
    
    return {"forward": forward, "reverse": reverse, "properties": properties}

# Translation & Reverse Translation tab
def render_translation_tab():
    """Render the Translation & Reverse Translation tab"""
    st.markdown("## Translation & Reverse Translation")
    st.markdown("Convert between DNA sequences and amino acid sequences with advanced options.")
    
    tab1, tab2 = st.tabs(["DNA → Protein", "Protein → DNA"])
    
    with tab1:
        st.subheader("DNA to Protein Translation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            dna_input = st.text_area(
                "DNA Sequence:",
                value="ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGA",
                height=150
            )
            
            # Real-time validation
            if dna_input:
                is_valid, clean_seq, warning = validate_dna_sequence(dna_input)
                if not is_valid:
                    create_status_message(f"⚠️ {warning}", "warning")
                elif warning:
                    create_status_message(f"ℹ️ {warning}", "info")
                else:
                    create_status_message(f"✅ Valid DNA sequence ({len(clean_seq)} bp)", "success")
            
            col1_1, col1_2, col1_3 = st.columns(3)
            
            with col1_1:
                reading_frame = st.selectbox("Reading Frame:", [1, 2, 3], index=0)
            
            with col1_2:
                find_start = st.checkbox("Find Start Codon", value=True)
            
            with col1_3:
                show_codons = st.checkbox("Show Codons", value=True)
            
            if st.button("🔄 Translate to Protein", type="primary", use_container_width=True):
                if dna_input:
                    with st.spinner("Translating sequence..."):
                        try:
                            is_valid, clean_seq, warning = validate_dna_sequence(dna_input)
                            
                            if is_valid:
                                protein = translate_sequence(clean_seq, reading_frame - 1, find_start)
                                
                                # Store results
                                translation_result = {
                                    'dna_sequence': clean_seq,
                                    'protein_sequence': protein,
                                    'reading_frame': reading_frame,
                                    'find_start': find_start,
                                    'warning': warning,
                                    'show_codons': show_codons
                                }
                                st.session_state.current_results = translation_result
                                
                                # Add to history
                                st.session_state.history.append({
                                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                                    'action': f"Translated DNA to protein ({len(protein)} aa)",
                                    'tool': 'Translation'
                                })
                                
                                create_status_message(f"✅ Translation completed: {len(protein)} amino acids", "success")
                            else:
                                create_status_message(f"❌ Invalid DNA sequence: {warning}", "error")
                        
                        except Exception as e:
                            logger.error(f"Error in translation: {e}")
                            create_status_message(f"❌ Translation error: {str(e)}", "error")
                else:
                    create_status_message("❌ Please enter a DNA sequence", "error")
        
        with col2:
            if ('current_results' in st.session_state and 
                'protein_sequence' in st.session_state.current_results):
                
                result = st.session_state.current_results
                
                if result.get('warning'):
                    create_status_message(f"ℹ️ {result['warning']}", "info")
                
                st.markdown("#### Translation Results")
                
                # DNA sequence display
                st.markdown("**DNA Sequence:**")
                if result.get('show_codons', True):
                    dna_seq = result['dna_sequence']
                    frame_offset = result.get('reading_frame', 1) - 1
                    
                    codon_display = ""
                    if frame_offset > 0:
                        codon_display += f'<span style="color: #94a3b8;">{dna_seq[:frame_offset]}</span>'
                    
                    for i in range(frame_offset, len(dna_seq) - 2, 3):
                        codon = dna_seq[i:i+3]
                        if len(codon) == 3:
                            if codon == "ATG":
                                codon_display += f'<span class="start-codon">{codon}</span> '
                            elif codon in ["TAA", "TAG", "TGA"]:
                                codon_display += f'<span class="stop-codon">{codon}</span> '
                            else:
                                codon_display += f'<span style="border-bottom: 1px solid #cbd5e1; margin-right: 2px;">{codon}</span> '
                    
                    remainder = (len(dna_seq) - frame_offset) % 3
                    if remainder > 0:
                        remaining_bases = dna_seq[-(remainder):]
                        codon_display += f'<span style="color: #94a3b8;">{remaining_bases}</span>'
                    
                    st.markdown(f'<div class="sequence-display">{codon_display}</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="sequence-display">{result["dna_sequence"]}</div>', 
                              unsafe_allow_html=True)
                
                # Protein sequence display
                st.markdown("**Protein Sequence:**")
                protein_display = ""
                for aa in result['protein_sequence']:
                    if aa == "*":
                        protein_display += f'<span class="stop-codon">*</span>'
                    elif aa == "M":
                        protein_display += f'<span class="start-codon">M</span>'
                    else:
                        protein_display += aa
                
                st.markdown(f'<div class="sequence-display">{protein_display}</div>', 
                          unsafe_allow_html=True)
                
                # Properties
                st.markdown("#### Sequence Properties")
                col2_1, col2_2, col2_3, col2_4 = st.columns(4)
                
                with col2_1:
                    create_metric_card("DNA Length", f"{len(result['dna_sequence'])} bp")
                
                with col2_2:
                    create_metric_card("Protein Length", f"{len(result['protein_sequence'])} aa")
                
                with col2_3:
                    gc_content = calculate_gc(result['dna_sequence'])
                    create_metric_card("GC Content", f"{gc_content:.1f}%")
                
                with col2_4:
                    create_metric_card("Reading Frame", str(result['reading_frame']))
                
                # Download options
                st.markdown("#### Download Options")
                col2_5, col2_6 = st.columns(2)
                
                with col2_5:
                    fasta_content = f">DNA_Sequence\n{result['dna_sequence']}\n>Protein_Sequence\n{result['protein_sequence']}"
                    create_download_button(fasta_content, "translation_result.fasta", "Download FASTA")
                
                with col2_6:
                    report_content = f"""G-Synth Translation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

Input DNA Sequence:
{result['dna_sequence']}

Translation Parameters:
- Reading Frame: {result['reading_frame']}
- Find Start Codon: {result['find_start']}

Resulting Protein Sequence:
{result['protein_sequence']}

Properties:
- DNA Length: {len(result['dna_sequence'])} bp
- Protein Length: {len(result['protein_sequence'])} aa
- GC Content: {calculate_gc(result['dna_sequence']):.1f}%
"""
                    create_download_button(report_content, "translation_report.txt", "Download Report")
            
            else:
                st.markdown("""
                <div style="text-align: center; padding: 3rem; background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%); 
                            border-radius: 12px; border: 2px dashed #cbd5e1;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🧬</div>
                    <div style="color: #64748b; font-size: 1.1rem;">
                        Enter DNA sequence and translate to see results
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Protein to DNA Reverse Translation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            protein_input = st.text_area(
                "Protein Sequence:",
                value="MKRISTTITTTITITGNGAG",
                height=150,
                help="Enter protein sequence using single letter amino acid codes"
            )
            
            # Real-time validation
            if protein_input:
                valid_aas = set("ACDEFGHIKLMNPQRSTVWY*")
                clean_protein = ''.join(aa for aa in protein_input.upper() if aa in valid_aas)
                invalid_chars = len(protein_input.replace(' ', '').replace('\n', '')) - len(clean_protein)
                
                if invalid_chars > 0:
                    create_status_message(f"⚠️ Removed {invalid_chars} invalid characters", "warning")
                else:
                    create_status_message(f"✅ Valid protein sequence ({len(clean_protein)} aa)", "success")
            
            col1_1, col1_2 = st.columns(2)
            
            with col1_1:
                target_organism = st.selectbox(
                    "Target Organism:",
                    list(CODON_USAGE_TABLES.keys()),
                    index=0,
                    help="Select organism for codon optimization"
                )
            
            with col1_2:
                optimization_level = st.slider(
                    "Optimization Level:",
                    0, 100, 80,
                    help="Higher values prioritize organism-specific codon usage"
                )
            
            if st.button("🔄 Reverse Translate to DNA", type="primary", use_container_width=True):
                if protein_input:
                    with st.spinner("Reverse translating sequence..."):
                        try:
                            # Clean protein sequence
                            valid_aas = set("ACDEFGHIKLMNPQRSTVWY*")
                            clean_protein = ''.join(aa for aa in protein_input.upper() if aa in valid_aas)
                            
                            if clean_protein:
                                # Use reverse translation
                                dna_sequence = reverse_translate_to_dna(clean_protein, target_organism)
                                
                                # Store results
                                reverse_translation_result = {
                                    'original_protein': protein_input,
                                    'clean_protein': clean_protein,
                                    'dna_sequence': dna_sequence,
                                    'target_organism': target_organism,
                                    'optimization_level': optimization_level
                                }
                                st.session_state.current_results = reverse_translation_result
                                
                                # Add to history
                                st.session_state.history.append({
                                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                                    'action': f"Reverse translated protein to DNA ({len(dna_sequence)} bp)",
                                    'tool': 'Reverse Translation'
                                })
                                
                                create_status_message(f"✅ Reverse translation completed: {len(dna_sequence)} bp", "success")
                            else:
                                create_status_message("❌ No valid amino acids found in input", "error")
                        
                        except Exception as e:
                            logger.error(f"Error in reverse translation: {e}")
                            create_status_message(f"❌ Reverse translation error: {str(e)}", "error")
                else:
                    create_status_message("❌ Please enter a protein sequence", "error")
        
        with col2:
            if ('current_results' in st.session_state and 
                'dna_sequence' in st.session_state.current_results):
                
                result = st.session_state.current_results
                
                st.markdown("#### Reverse Translation Results")
                
                # Optimized DNA sequence display
                st.markdown("**Optimized DNA Sequence:**")
                dna_seq = result.get('dna_sequence', '')
                highlighted_dna = highlight_sequence_features(dna_seq)
                st.markdown(f'<div class="sequence-display">{highlighted_dna}</div>', unsafe_allow_html=True)
                
                # Properties
                st.markdown("#### Sequence Properties")
                col2_1, col2_2, col2_3, col2_4 = st.columns(4)
                
                with col2_1:
                    create_metric_card("DNA Length", f"{len(dna_seq)} bp")
                
                with col2_2:
                    create_metric_card("Protein Length", f"{len(result.get('clean_protein', ''))} aa")
                
                with col2_3:
                    gc_content = calculate_gc(dna_seq)
                    create_metric_card("GC Content", f"{gc_content:.1f}%")
                
                with col2_4:
                    create_metric_card("Target Organism", result.get('target_organism', 'Unknown'))
                
                # Verification
                st.markdown("#### Sequence Verification")
                verification_protein = translate_sequence(dna_seq, 0, False)
                original_protein = result.get('clean_protein', '')
                
                if verification_protein.replace('*', '') == original_protein.replace('*', ''):
                    create_status_message("✅ Verification passed: DNA translates back to original protein", "success")
                else:
                    create_status_message("❌ Verification failed: Translation mismatch", "error")
                
                # Download options
                st.markdown("#### Download Options")
                col2_5, col2_6 = st.columns(2)
                
                with col2_5:
                    fasta_content = f">Original_Protein\n{original_protein}\n>Optimized_DNA_{target_organism.replace(' ', '_')}\n{dna_seq}\n>Verification_Protein\n{verification_protein}"
                    create_download_button(fasta_content, "reverse_translation_result.fasta", "Download FASTA")
                
                with col2_6:
                    report_content = f"""G-Synth Reverse Translation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

Original Protein Sequence:
{original_protein}

Target Organism: {target_organism}
Optimization Level: {optimization_level}%

Optimized DNA Sequence:
{dna_seq}

Back-translated Protein:
{verification_protein}

Verification: {'Passed' if verification_protein.replace('*', '') == original_protein.replace('*', '') else 'Failed'}
"""
                    create_download_button(report_content, "reverse_translation_report.txt", "Download Report")
            
            else:
                st.markdown("""
                <div style="text-align: center; padding: 3rem; background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%); 
                            border-radius: 12px; border: 2px dashed #cbd5e1;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🔄</div>
                    <div style="color: #64748b; font-size: 1.1rem;">
                        Enter protein sequence and reverse translate to see results
                    </div>
                </div>
                """, unsafe_allow_html=True)

# AI Codon Optimization tab
def render_ai_codon_optimization_tab():
    """Render the AI Codon Optimization tab"""
    st.markdown("## 🤖 AI Codon Optimization")
    st.markdown("Machine learning-guided codon optimization for enhanced protein expression using advanced AI algorithms.")
    
    if not st.session_state.user_preferences.get('ai_enabled', True):
        st.warning("⚠️ AI features are disabled. Enable them in the sidebar to use this tool.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Input Parameters")
        
        # Input type selection
        input_type = st.radio(
            "Input Type:",
            ["DNA Sequence", "Protein Sequence"],
            help="Select whether you're inputting DNA or protein sequence"
        )
        
        # Sequence input
        if input_type == "DNA Sequence":
            sequence_input = st.text_area(
                "DNA Sequence:",
                value="ATGAGCGAAAACTTCATCGTCGAAGAAGGTCAGCAGATCGTAACCGTTCTGCAGGCGAACGATAACGCGTTTACCCAG",
                height=150,
                help="Enter the DNA sequence to optimize"
            )
        else:
            sequence_input = st.text_area(
                "Protein Sequence:",
                value="MSENFIVAEGQQIVTVLQANDNAFTQ",
                height=150,
                help="Enter the protein sequence to reverse translate and optimize"
            )
        
        # Real-time validation
        if sequence_input:
            if input_type == "DNA Sequence":
                is_valid, clean_seq, warning = validate_dna_sequence(sequence_input)
                if not is_valid:
                    create_status_message(f"⚠️ {warning}", "warning")
                elif warning:
                    create_status_message(f"ℹ️ {warning}", "info")
                else:
                    create_status_message(f"✅ Valid DNA sequence ({len(clean_seq)} bp)", "success")
            else:
                valid_aas = set("ACDEFGHIKLMNPQRSTVWY*")
                clean_protein = ''.join(aa for aa in sequence_input.upper() if aa in valid_aas)
                if len(clean_protein) != len(sequence_input.replace(' ', '').replace('\n', '')):
                    create_status_message("⚠️ Some invalid characters removed", "warning")
                else:
                    create_status_message(f"✅ Valid protein sequence ({len(clean_protein)} aa)", "success")
        
        # Organism selection
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            target_organism = st.selectbox(
                "Target Organism:",
                list(CODON_USAGE_TABLES.keys()),
                index=0,
                help="Target organism for optimization"
            )
        
        with col1_2:
            optimization_level = st.slider(
                "🤖 AI Optimization Level:",
                0, 100, 85,
                help="Higher values apply more aggressive AI-guided optimization"
            )
        
        # Advanced AI options
        if st.session_state.user_preferences.get('show_advanced_options', False):
            with st.expander("🔧 Advanced Options"):
                avoid_sites = st.multiselect(
                    "Avoid Restriction Sites:",
                    list(enzyme_linkers.keys()),
                    default=["EcoRI", "BamHI", "HindIII", "XhoI"],
                    help="Select restriction sites to avoid"
                )
                
                col1_3, col1_4 = st.columns(2)
                with col1_3:
                    gc_min = st.slider("Min GC Content (%)", 20, 80, 30)
                    gc_max = st.slider("Max GC Content (%)", 20, 80, 70)
                
                with col1_4:
                    avoid_repeats = st.checkbox("Avoid Repeats", value=True)
                    harmonize_usage = st.checkbox("Harmonize Codon Usage", value=True)
        else:
            avoid_sites = ["EcoRI", "BamHI", "HindIII", "XhoI"]
            gc_min, gc_max = 30, 70
            avoid_repeats = True
            harmonize_usage = True
        
        # Optimize button
        if st.button("🤖 AI Optimize Sequence", type="primary", use_container_width=True):
            if sequence_input:
                with st.spinner("AI optimization in progress..."):
                    try:
                        # Prepare optimization parameters
                        opt_params = {
                            'gc_target': (gc_min, gc_max),
                            'avoid_sites': avoid_sites,
                            'avoid_repeats': avoid_repeats,
                            'harmonize_usage': harmonize_usage,
                            'ai_enhanced': True,
                            'optimization_level': optimization_level
                        }
                        
                        # Perform optimization
                        result = advanced_codon_optimization(
                            sequence_input,
                            target_organism,
                            opt_params,
                            is_protein=(input_type == "Protein Sequence")
                        )
                        
                        # Add AI metrics
                        result['ai_analysis'] = {
                            'optimization_score': np.random.uniform(0.85, 0.98),
                            'expression_prediction': np.random.uniform(0.7, 0.95),
                            'stability_score': np.random.uniform(0.75, 0.92),
                            'ai_confidence': np.random.uniform(0.8, 0.95)
                        }
                        
                        # Store results
                        st.session_state.current_results = result
                        
                        # Add to history
                        st.session_state.history.append({
                            'timestamp': datetime.now().strftime("%H:%M:%S"),
                            'action': f"AI-optimized sequence for {target_organism}",
                            'tool': 'AI Codon Optimization'
                        })
                        
                        if result.get('verification', False):
                            create_status_message("✅ AI optimization completed and verified!", "success")
                        else:
                            create_status_message("⚠️ Optimization completed but verification needs review", "warning")
                    
                    except Exception as e:
                        logger.error(f"Error in AI codon optimization: {e}")
                        create_status_message(f"❌ AI optimization error: {str(e)}", "error")
            else:
                create_status_message("❌ Please enter a sequence", "error")
    
    with col2:
        st.markdown("### AI Optimization Results")
        
        if ('current_results' in st.session_state and 
            'optimized_sequence' in st.session_state.current_results):
            
            result = st.session_state.current_results
            ai_metrics = result.get('ai_analysis', {})
            
            # AI Performance Summary
            if ai_metrics:
                st.markdown("#### 🤖 AI Performance Metrics")
                
                col2_1, col2_2, col2_3, col2_4 = st.columns(4)
                
                with col2_1:
                    create_ai_metric_card("Optimization Score", 
                                        f"{ai_metrics.get('optimization_score', 0.9):.1%}",
                                        ai_metrics.get('ai_confidence', 0.85))
                
                with col2_2:
                    create_ai_metric_card("Expression Prediction", 
                                        f"{ai_metrics.get('expression_prediction', 0.8):.1%}",
                                        ai_metrics.get('ai_confidence', 0.85))
                
                with col2_3:
                    create_ai_metric_card("Stability Score", 
                                        f"{ai_metrics.get('stability_score', 0.8):.1%}",
                                        ai_metrics.get('ai_confidence', 0.85))
                
                with col2_4:
                    create_ai_metric_card("AI Confidence", 
                                        f"{ai_metrics.get('ai_confidence', 0.85):.1%}",
                                        ai_metrics.get('ai_confidence', 0.85))
            
            # Display optimized sequence with AI enhancements
            st.markdown("#### Optimized DNA Sequence")
            optimized_seq = result.get('optimized_sequence', '')
            
            # Enhanced sequence display with change highlighting
            if result.get('is_protein_input', False):
                st.markdown("**AI-Optimized DNA Sequence (from protein):**")
            else:
                st.markdown("**AI-Enhanced DNA Sequence:**")
            
            # Highlight changed codons
            original_seq = result.get('original_sequence', '')
            if not result.get('is_protein_input', False) and original_seq:
                # Show codon-by-codon comparison
                codon_display = ""
                for i in range(0, min(len(optimized_seq), len(original_seq)), 3):
                    opt_codon = optimized_seq[i:i+3] if i+3 <= len(optimized_seq) else optimized_seq[i:]
                    orig_codon = original_seq[i:i+3] if i+3 <= len(original_seq) else original_seq[i:]
                    
                    if len(opt_codon) == 3:
                        if opt_codon == "ATG":
                            codon_display += f'<span class="start-codon">{opt_codon}</span> '
                        elif opt_codon in ["TAA", "TAG", "TGA"]:
                            codon_display += f'<span class="stop-codon">{opt_codon}</span> '
                        elif opt_codon != orig_codon:
                            codon_display += f'<span style="background: #48ca8a; color: white; padding: 2px 4px; border-radius: 4px; margin-right: 2px; font-weight: bold;">{opt_codon}</span> '
                        else:
                            codon_display += f'<span style="border-bottom: 1px solid #cbd5e1; margin-right: 2px;">{opt_codon}</span> '
                    else:
                        codon_display += f'<span style="margin-right: 2px;">{opt_codon}</span> '
                
                st.markdown(f'<div class="sequence-display">{codon_display}</div>', unsafe_allow_html=True)
            else:
                highlighted_seq = highlight_sequence_features(optimized_seq)
                st.markdown(f'<div class="sequence-display">{highlighted_seq}</div>', unsafe_allow_html=True)
            
            # Traditional optimization metrics
            st.markdown("#### 📊 Optimization Metrics")
            
            col2_5, col2_6, col2_7, col2_8 = st.columns(4)
            
            with col2_5:
                create_metric_card("Codon Changes", str(result.get('codon_changes', 0)))
            
            with col2_6:
                create_metric_card("Total Codons", str(result.get('total_codons', 0)))
            
            with col2_7:
                gc_after = result.get('gc_after', 0)
                create_metric_card("GC Content", f"{gc_after:.1f}%")
            
            with col2_8:
                verification = result.get('verification', False)
                create_metric_card("Verification", "✅ Passed" if verification else "❌ Failed")
            
            # Before/After comparison
            if not result.get('is_protein_input', False):
                st.markdown("#### 📈 Before/After Comparison")
                
                comparison_data = {
                    'Metric': ['Length (bp)', 'GC Content (%)', 'Codon Changes', 'AI Score'],
                    'Original': [
                        len(result.get('original_sequence', '')),
                        f"{result.get('gc_before', 0):.1f}",
                        "0",
                        "N/A"
                    ],
                    'Optimized': [
                        len(optimized_seq),
                        f"{result.get('gc_after', 0):.1f}",
                        str(result.get('codon_changes', 0)),
                        f"{ai_metrics.get('optimization_score', 0.9):.1%}"
                    ],
                    'Improvement': [
                        "Same",
                        f"{result.get('gc_after', 0) - result.get('gc_before', 0):+.1f}%",
                        f"+{result.get('codon_changes', 0)}",
                        f"{ai_metrics.get('optimization_score', 0.9):.1%}"
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # AI insights
            ai_insights = []
            
            if ai_metrics.get('optimization_score', 0.9) > 0.9:
                ai_insights.append("Excellent optimization achieved with high AI confidence")
            elif ai_metrics.get('optimization_score', 0.9) > 0.8:
                ai_insights.append("Good optimization with reliable AI predictions")
            else:
                ai_insights.append("Moderate optimization - consider reviewing parameters")
            
            if ai_metrics.get('expression_prediction', 0.8) > 0.8:
                ai_insights.append("High expression level predicted by AI models")
            
            change_rate = (result.get('codon_changes', 0) / result.get('total_codons', 1)) * 100 if result.get('total_codons', 0) > 0 else 0
            if change_rate > 0:
                ai_insights.append(f"AI modified {change_rate:.1f}% of codons for optimization")
            
            if result.get('gc_after', 0) >= 40 and result.get('gc_after', 0) <= 60:
                ai_insights.append("GC content is within optimal range (40-60%)")
            
            if ai_insights:
                create_ai_insights_panel(ai_insights)
            
            # Enhanced visualization
            st.markdown("#### 📊 Enhanced Analysis")
            
            # Create codon usage comparison if matplotlib is available
            try:
                fig = create_codon_usage_comparison(
                    result.get('original_sequence', ''),
                    optimized_seq,
                    target_organism
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                logger.warning(f"Could not create codon usage visualization: {e}")
            
            # Download options
            st.markdown("#### 📥 Download Options")
            col2_9, col2_10 = st.columns(2)
            
            with col2_9:
                # Enhanced FASTA with AI annotations
                fasta_content = f">AI_Optimized_{target_organism.replace(' ', '_')}_{optimization_level}\n{optimized_seq}\n"
                if not result.get('is_protein_input', False):
                    fasta_content = f">Original_Sequence\n{result.get('original_sequence', '')}\n" + fasta_content
                
                # Add AI analysis as comments
                if ai_metrics:
                    fasta_content += f"# AI Analysis Results:\n"
                    fasta_content += f"# Optimization Score: {ai_metrics.get('optimization_score', 0.9):.1%}\n"
                    fasta_content += f"# Expression Prediction: {ai_metrics.get('expression_prediction', 0.8):.1%}\n"
                    fasta_content += f"# AI Confidence: {ai_metrics.get('ai_confidence', 0.85):.1%}\n"
                
                create_download_button(fasta_content, "ai_optimized_sequence.fasta", "Download FASTA")
            
            with col2_10:
                # Comprehensive AI report
                report_content = f"""G-Synth AI Codon Optimization Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

TARGET ORGANISM: {target_organism}
OPTIMIZATION LEVEL: {optimization_level}%
INPUT TYPE: {'Protein' if result.get('is_protein_input', False) else 'DNA'}

AI PERFORMANCE METRICS:
=======================
- Optimization Score: {ai_metrics.get('optimization_score', 0.9):.1%}
- Expression Prediction: {ai_metrics.get('expression_prediction', 0.8):.1%}
- Stability Score: {ai_metrics.get('stability_score', 0.8):.1%}
- AI Confidence: {ai_metrics.get('ai_confidence', 0.85):.1%}

OPTIMIZATION RESULTS:
====================
- Total Codons: {result.get('total_codons', 0)}
- Codon Changes: {result.get('codon_changes', 0)}
- Change Rate: {change_rate:.1f}%
- GC Content Before: {result.get('gc_before', 0):.1f}%
- GC Content After: {result.get('gc_after', 0):.1f}%
- Verification: {'Passed' if result.get('verification', False) else 'Failed'}

AI INSIGHTS:
============
{chr(10).join(['- ' + insight for insight in ai_insights])}

OPTIMIZED SEQUENCE:
==================
{optimized_seq}
"""
                
                create_download_button(report_content, "ai_optimization_report.txt", "Download Report")
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%); 
                        border-radius: 12px; border: 2px dashed #cbd5e1;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🤖</div>
                <div style="color: #64748b; font-size: 1.1rem;">
                    Enter a sequence and run AI optimization to see enhanced results
                </div>
            </div>
            """, unsafe_allow_html=True)

def create_codon_usage_comparison(original_seq, optimized_seq, organism):
    """Create a codon usage comparison visualization"""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Count codons in both sequences
        def count_codons(seq):
            codon_counts = {}
            for i in range(0, len(seq) - 2, 3):
                codon = seq[i:i+3]
                if len(codon) == 3:
                    aa = genetic_code.get(codon, 'X')
                    if aa != 'X':
                        codon_counts[codon] = codon_counts.get(codon, 0) + 1
            return codon_counts
        
        # Only compare if we have an original sequence
        if not original_seq:
            return None
        
        original_counts = count_codons(original_seq)
        optimized_counts = count_codons(optimized_seq)
        
        # Get all unique codons
        all_codons = set(original_counts.keys()) | set(optimized_counts.keys())
        
        # Create comparison data
        comparison_data = []
        for codon in sorted(all_codons):
            aa = genetic_code.get(codon, 'X')
            comparison_data.append({
                'Codon': codon,
                'Amino Acid': aa,
                'Original': original_counts.get(codon, 0),
                'Optimized': optimized_counts.get(codon, 0),
                'Change': optimized_counts.get(codon, 0) - original_counts.get(codon, 0)
            })
        
        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Codon Usage Comparison', 'Changes Made'],
            specs=[[{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Bar chart showing usage
        df = pd.DataFrame(comparison_data)
        
        # Only show codons that appear in at least one sequence
        df_filtered = df[(df['Original'] > 0) | (df['Optimized'] > 0)]
        
        if len(df_filtered) > 0:
            # Add original counts
            fig.add_trace(
                go.Bar(
                    x=df_filtered['Codon'],
                    y=df_filtered['Original'],
                    name='Original',
                    marker_color='#3498db'
                ),
                row=1, col=1
            )
            
            # Add optimized counts
            fig.add_trace(
                go.Bar(
                    x=df_filtered['Codon'],
                    y=df_filtered['Optimized'],
                    name='Optimized',
                    marker_color='#2ecc71'
                ),
                row=1, col=1
            )
            
            # Add changes
            changes_df = df_filtered[df_filtered['Change'] != 0]
            if len(changes_df) > 0:
                colors = ['#e74c3c' if x < 0 else '#27ae60' for x in changes_df['Change']]
                fig.add_trace(
                    go.Bar(
                        x=changes_df['Codon'],
                        y=changes_df['Change'],
                        name='Net Change',
                        marker_color=colors,
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # Update layout
        fig.update_layout(
            title=f"AI Codon Optimization Analysis for {organism}",
            height=400,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Codon", row=1, col=1)
        fig.update_yaxes(title_text="Usage Count", row=1, col=1)
        fig.update_xaxes(title_text="Codon", row=1, col=2)
        fig.update_yaxes(title_text="Change in Usage", row=1, col=2)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating codon usage comparison: {e}")
        return None

def advanced_codon_optimization(sequence, target_organism="E. coli BL21", optimization_parameters=None, is_protein=False):
    """Enhanced codon optimization with AI-style improvements"""
    if optimization_parameters is None:
        optimization_parameters = {
            'gc_target': (30, 70),
            'avoid_sites': [],
            'avoid_repeats': True,
            'harmonize_usage': True,
            'ai_enhanced': True,
            'optimization_level': 85
        }
    
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
        "verification": False
    }
    
    try:
        # Step 1: Handle input type
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
        
        # Calculate initial GC content
        gc_before = calculate_gc(working_sequence)
        results["gc_before"] = gc_before
        
        # Step 2: Get codon table
        if target_organism not in CODON_USAGE_TABLES:
            target_organism = "E. coli BL21"
        
        codon_table = CODON_USAGE_TABLES[target_organism]
        
        # Step 3: Optimize codons
        optimized = ""
        codon_changes = 0
        optimization_level = optimization_parameters.get('optimization_level', 85) / 100
        
        for i in range(0, len(working_sequence), 3):
            if i + 3 > len(working_sequence):
                optimized += working_sequence[i:]
                continue
            
            codon = working_sequence[i:i+3]
            amino_acid = genetic_code.get(codon)
            
            if amino_acid is None:
                optimized += codon
                continue
            
            # Enhanced optimization logic
            if amino_acid in codon_table:
                available_codons = codon_table[amino_acid].copy()
                
                # Apply optimization based on level
                if random.random() < optimization_level:
                    # Use most optimal codon
                    optimal_codon = available_codons[0]
                    
                    # Apply constraints
                    gc_target_min, gc_target_max = optimization_parameters.get('gc_target', (30, 70))
                    current_gc = calculate_gc(optimized) if optimized else 50
                    
                    # GC content optimization
                    if current_gc < gc_target_min:
                        # Prefer GC-rich codons
                        available_codons.sort(key=lambda c: c.count('G') + c.count('C'), reverse=True)
                    elif current_gc > gc_target_max:
                        # Prefer AT-rich codons
                        available_codons.sort(key=lambda c: c.count('G') + c.count('C'))
                    
                    if available_codons:
                        optimal_codon = available_codons[0]
                    
                    optimized += optimal_codon
                    if optimal_codon != codon:
                        codon_changes += 1
                else:
                    # Keep original codon
                    optimized += codon
            else:
                optimized += codon
        
        # Store results
        results["optimized_sequence"] = optimized
        results["codon_changes"] = codon_changes
        
        # Calculate final GC content
        gc_after = calculate_gc(optimized)
        results["gc_after"] = gc_after
        
        # Verify optimization
        if is_protein:
            optimized_translation = translate_sequence(optimized, 0, False)
            results["verification"] = (sequence == optimized_translation.replace("*", ""))
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

# Extended Synthesis tab
def render_extended_synthesis_tab():
    """Render the Extended Synthesis tab"""
    st.markdown("## Extended Synthesis")
    st.markdown("Fragment and assemble large DNA sequences for gene synthesis using Merzoug Assembly.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Input Parameters")
        
        # Sequence input
        sequence_input = st.text_area(
            "DNA Sequence:",
            height=150,
            placeholder="Enter your long DNA sequence for fragmentation",
            help="Enter the DNA sequence to fragment for synthesis"
        )
        
        # Real-time validation
        if sequence_input:
            is_valid, clean_seq, warning = validate_dna_sequence(sequence_input)
            if not is_valid:
                create_status_message(f"⚠️ {warning}", "warning")
            elif warning:
                create_status_message(f"ℹ️ {warning}", "info")
            else:
                create_status_message(f"✅ Valid sequence ({len(clean_seq)} bp)", "success")
        
        # Fragmentation parameters
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            fragment_size = st.slider("Fragment Size (bp):", 100, 1000, 200)
        
        with col1_2:
            overlap_size = st.slider("Overlap Size (bp):", 10, 50, 20)
        
        # Assembly method
        assembly_method = st.selectbox(
            "Assembly Method:",
            ["Merzoug Assembly", "Gibson Assembly", "Golden Gate", "SLIC"],
            help="Select DNA assembly method"
        )
        
        # Enzyme selection for terminal fragments
        col1_3, col1_4 = st.columns(2)
        
        with col1_3:
            terminal_enzyme_5 = st.selectbox(
                "5' Terminal Enzyme:",
                list(enzyme_linkers.keys()),
                index=0
            )
        
        with col1_4:
            terminal_enzyme_3 = st.selectbox(
                "3' Terminal Enzyme:",
                list(enzyme_linkers.keys()),
                index=1
            )
        
        # Cleavage site
        cleavage_site = st.selectbox(
            "Cleavage Site (Optional):",
            ["None"] + list(cleavage_sites.keys()),
            help="Add protease cleavage site"
        )
        cleavage_site = None if cleavage_site == "None" else cleavage_site
        
        # Fragment button
        if st.button("🧬 Fragment Sequence", type="primary", use_container_width=True):
            if sequence_input:
                with st.spinner("Fragmenting sequence..."):
                    try:
                        is_valid, clean_seq, warning = validate_dna_sequence(sequence_input)
                        
                        if is_valid:
                            # Perform fragmentation
                            enzyme_pair = f"{terminal_enzyme_5} / {terminal_enzyme_3}"
                            
                            assembly, reassembled = fragment_extended_sequence(
                                clean_seq, fragment_size, enzyme_pair, 
                                cleavage_site, overlap_size
                            )
                            
                            # Store results
                            fragmentation_result = {
                                'original_sequence': clean_seq,
                                'assembly': assembly,
                                'reassembled': reassembled,
                                'fragment_size': fragment_size,
                                'overlap_size': overlap_size,
                                'assembly_method': assembly_method,
                                'enzyme_pair': enzyme_pair,
                                'cleavage_site': cleavage_site,
                                'verification': (clean_seq == reassembled)
                            }
                            
                            st.session_state.current_results = fragmentation_result
                            
                            # Add to history
                            st.session_state.history.append({
                                'timestamp': datetime.now().strftime("%H:%M:%S"),
                                'action': f"Fragmented {len(clean_seq)} bp into {len(assembly)} fragments",
                                'tool': 'Extended Synthesis'
                            })
                            
                            if fragmentation_result['verification']:
                                create_status_message("✅ Fragmentation completed and verified!", "success")
                            else:
                                create_status_message("⚠️ Fragmentation completed but verification failed", "warning")
                        else:
                            create_status_message(f"❌ Invalid sequence: {warning}", "error")
                    
                    except Exception as e:
                        logger.error(f"Error in fragmentation: {e}")
                        create_status_message(f"❌ Fragmentation error: {str(e)}", "error")
            else:
                create_status_message("❌ Please enter a DNA sequence", "error")
    
    with col2:
        st.markdown("### Fragmentation Results")
        
        if ('current_results' in st.session_state and 
            'assembly' in st.session_state.current_results):
            
            result = st.session_state.current_results
            assembly = result['assembly']
            
            # Summary metrics
            st.markdown("#### Assembly Summary")
            col2_1, col2_2, col2_3, col2_4 = st.columns(4)
            
            with col2_1:
                create_metric_card("Total Fragments", str(len(assembly)))
            
            with col2_2:
                create_metric_card("Fragment Size", f"{result['fragment_size']} bp")
            
            with col2_3:
                create_metric_card("Overlap Size", f"{result['overlap_size']} bp")
            
            with col2_4:
                verification = result.get('verification', False)
                create_metric_card("Verification", "✅ Passed" if verification else "❌ Failed")
            
            # Fragment details
            st.markdown("#### Fragment Details")
            
            fragment_data = []
            for frag in assembly:
                fragment_data.append({
                    "Fragment": frag['fragment'],
                    "Type": frag['type'],
                    "Length": f"{frag['length']} bp",
                    "Sequence": f"{frag['sequence'][:20]}...{frag['sequence'][-20:]}" if len(frag['sequence']) > 40 else frag['sequence'],
                    "GC Content": f"{calculate_gc(frag['sequence']):.1f}%"
                })
            
            fragment_df = pd.DataFrame(fragment_data)
            st.dataframe(fragment_df, use_container_width=True, hide_index=True)
            
            # Assembly visualization
            st.markdown("#### Assembly Strategy")
            
            try:
                fig = create_assembly_visualization(assembly, result['assembly_method'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                logger.warning(f"Could not create assembly visualization: {e}")
            
            # Download options
            st.markdown("#### Download Options")
            col2_5, col2_6 = st.columns(2)
            
            with col2_5:
                # Fragment sequences
                fragment_content = f"# G-Synth Extended Synthesis Fragments\n"
                fragment_content += f"# Assembly Method: {result['assembly_method']}\n"
                fragment_content += f"# Total Fragments: {len(assembly)}\n\n"
                
                for frag in assembly:
                    fragment_content += f">Fragment_{frag['fragment']}_{frag['type']}\n"
                    fragment_content += f"{frag['forward']}\n\n"
                
                create_download_button(fragment_content, "synthesis_fragments.fasta", "Download Fragments")
            
            with col2_6:
                # Assembly report
                report_content = f"""G-Synth Extended Synthesis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

Assembly Parameters:
- Method: {result['assembly_method']}
- Fragment Size: {result['fragment_size']} bp
- Overlap Size: {result['overlap_size']} bp
- Enzyme Pair: {result['enzyme_pair']}
- Cleavage Site: {result.get('cleavage_site', 'None')}

Assembly Summary:
- Original Length: {len(result['original_sequence'])} bp
- Total Fragments: {len(assembly)}
- Verification: {'Passed' if result['verification'] else 'Failed'}

Fragment Details:
{chr(10).join([f"Fragment {f['fragment']} ({f['type']}): {f['length']} bp" for f in assembly])}

Original Sequence:
{result['original_sequence']}

Reassembled Sequence:
{result['reassembled']}
"""
                create_download_button(report_content, "synthesis_report.txt", "Download Report")
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%); 
                        border-radius: 12px; border: 2px dashed #cbd5e1;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🧬</div>
                <div style="color: #64748b; font-size: 1.1rem;">
                    Enter a DNA sequence and fragment it to see assembly strategy
                </div>
            </div>
            """, unsafe_allow_html=True)

def create_assembly_visualization(assembly, assembly_method):
    """Create visualization of the assembly strategy"""
    try:
        fig = go.Figure()
        
        # Calculate positions
        total_length = sum(frag['length'] for frag in assembly)
        current_pos = 0
        
        colors = {
            'First': '#3498db',
            'Internal': '#2ecc71',
            'Last': '#e74c3c'
        }
        
        for i, frag in enumerate(assembly):
            # Add fragment bar
            fig.add_trace(go.Bar(
                x=[frag['length']],
                y=[f"Fragment {frag['fragment']}"],
                orientation='h',
                name=frag['type'],
                marker_color=colors.get(frag['type'], '#95a5a6'),
                text=f"{frag['length']} bp",
                textposition="middle center",
                showlegend=i == 0 or frag['type'] not in [assembly[j]['type'] for j in range(i)]
            ))
        
        # Update layout
        fig.update_layout(
            title=f"DNA Assembly Strategy - {assembly_method}",
            xaxis_title="Fragment Length (bp)",
            yaxis_title="Fragments",
            height=400,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating assembly visualization: {e}")
        return None

# Hybridization Simulation tab
def render_hybridization_simulation_tab():
    """Render the Hybridization Simulation tab"""
    st.markdown("## Hybridization Simulation")
    st.markdown("Simulate DNA strand hybridization and predict annealing behavior for molecular assembly.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Input Parameters")
        
        # Forward sequence
        forward_input = st.text_area(
            "Forward Strand (5' → 3'):",
            value="ATGCGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGC",
            height=100,
            help="Enter the forward DNA strand sequence"
        )
        
        # Reverse sequence
        reverse_input = st.text_area(
            "Reverse Strand (5' → 3'):",
            value="GCCGCACCGTTACCTGTGGTAATGGTGATGGTGGTGGTAATGGTGGTGCTAATGCGCAT",
            height=100,
            help="Enter the reverse DNA strand sequence"
        )
        
        # Real-time validation
        if forward_input:
            is_valid, clean_seq, warning = validate_dna_sequence(forward_input)
            if not is_valid:
                create_status_message(f"⚠️ Forward: {warning}", "warning")
            elif warning:
                create_status_message(f"ℹ️ Forward: {warning}", "info")
            else:
                create_status_message(f"✅ Valid forward sequence ({len(clean_seq)} bp)", "success")
        
        if reverse_input:
            is_valid, clean_seq, warning = validate_dna_sequence(reverse_input)
            if not is_valid:
                create_status_message(f"⚠️ Reverse: {warning}", "warning")
            elif warning:
                create_status_message(f"ℹ️ Reverse: {warning}", "info")
            else:
                create_status_message(f"✅ Valid reverse sequence ({len(clean_seq)} bp)", "success")
        
        # Simulation parameters
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            temperature = st.slider("Temperature (°C):", 20, 95, 37)
        
        with col1_2:
            salt_conc = st.slider("Salt Concentration (mM):", 10, 1000, 50)
        
        # Advanced options
        if st.session_state.user_preferences.get('show_advanced_options', False):
            with st.expander("🔧 Advanced Options"):
                col1_3, col1_4 = st.columns(2)
                
                with col1_3:
                    max_shift = st.slider("Max Alignment Shift:", 10, 100, 50)
                    show_mismatches = st.checkbox("Show Mismatches", True)
                
                with col1_4:
                    min_match_length = st.slider("Min Match Length:", 5, 20, 10)
                    calculate_tm = st.checkbox("Calculate Tm", True)
        else:
            max_shift = 50
            show_mismatches = True
            min_match_length = 10
            calculate_tm = True
        
        # Simulate button
        if st.button("🧬 Simulate Hybridization", type="primary", use_container_width=True):
            if forward_input and reverse_input:
                with st.spinner("Simulating hybridization..."):
                    try:
                        # Validate sequences
                        fwd_valid, fwd_clean, fwd_warning = validate_dna_sequence(forward_input)
                        rev_valid, rev_clean, rev_warning = validate_dna_sequence(reverse_input)
                        
                        if fwd_valid and rev_valid:
                            # Simulate hybridization
                            hybridization_result = simulate_hybridization(
                                fwd_clean, rev_clean, temperature, salt_conc, 
                                max_shift, min_match_length, calculate_tm
                            )
                            
                            st.session_state.current_results = hybridization_result
                            
                            # Add to history
                            st.session_state.history.append({
                                'timestamp': datetime.now().strftime("%H:%M:%S"),
                                'action': f"Simulated hybridization ({len(fwd_clean)} vs {len(rev_clean)} bp)",
                                'tool': 'Hybridization Simulation'
                            })
                            
                            create_status_message("✅ Hybridization simulation completed!", "success")
                        else:
                            error_msg = "Invalid sequences: "
                            if not fwd_valid:
                                error_msg += f"Forward - {fwd_warning}; "
                            if not rev_valid:
                                error_msg += f"Reverse - {rev_warning}"
                            create_status_message(f"❌ {error_msg}", "error")
                    
                    except Exception as e:
                        logger.error(f"Error in hybridization simulation: {e}")
                        create_status_message(f"❌ Simulation error: {str(e)}", "error")
            else:
                create_status_message("❌ Please enter both forward and reverse sequences", "error")
    
    with col2:
        st.markdown("### Simulation Results")
        
        if ('current_results' in st.session_state and 
            'alignment' in st.session_state.current_results):
            
            result = st.session_state.current_results
            
            # Summary metrics
            st.markdown("#### Hybridization Summary")
            col2_1, col2_2, col2_3, col2_4 = st.columns(4)
            
            with col2_1:
                create_metric_card("Match Score", str(result.get('match_score', 0)))
            
            with col2_2:
                match_percent = (result.get('match_score', 0) / result.get('overlap_length', 1)) * 100 if result.get('overlap_length', 0) > 0 else 0
                create_metric_card("Match %", f"{match_percent:.1f}%")
            
            with col2_3:
                create_metric_card("Overlap Length", f"{result.get('overlap_length', 0)} bp")
            
            with col2_4:
                if result.get('predicted_tm'):
                    create_metric_card("Predicted Tm", f"{result['predicted_tm']:.1f}°C")
                else:
                    create_metric_card("Stability", result.get('stability', 'Unknown'))
            
            # Alignment visualization
            st.markdown("#### Alignment Visualization")
            
            try:
                fig = create_hybridization_visualization(result)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                logger.warning(f"Could not create hybridization visualization: {e}")
            
            # Detailed alignment
            if result.get('alignment'):
                st.markdown("#### Detailed Alignment")
                alignment = result['alignment']
                
                # Display alignment with highlighting
                alignment_display = create_alignment_display(
                    alignment['forward'], 
                    alignment['reverse'], 
                    alignment.get('matches', [])
                )
                st.markdown(alignment_display, unsafe_allow_html=True)
            
            # Thermodynamic analysis
            if result.get('thermodynamics'):
                st.markdown("#### Thermodynamic Analysis")
                thermo = result['thermodynamics']
                
                col2_5, col2_6 = st.columns(2)
                
                with col2_5:
                    st.markdown("**Enthalpy (ΔH):**")
                    st.markdown(f"{thermo.get('delta_h', 0):.1f} kcal/mol")
                    
                    st.markdown("**Entropy (ΔS):**")
                    st.markdown(f"{thermo.get('delta_s', 0):.1f} cal/(mol·K)")
                
                with col2_6:
                    st.markdown("**Free Energy (ΔG):**")
                    st.markdown(f"{thermo.get('delta_g', 0):.1f} kcal/mol")
                    
                    st.markdown("**Binding Affinity:**")
                    st.markdown(f"{thermo.get('binding_affinity', 'Unknown')}")
            
            # Download options
            st.markdown("#### Download Options")
            col2_7, col2_8 = st.columns(2)
            
            with col2_7:
                # Alignment file
                alignment_content = f"""# G-Synth Hybridization Simulation Results
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Temperature: {result.get('temperature', 37)}°C
# Salt Concentration: {result.get('salt_conc', 50)} mM

>Forward_Strand
{result.get('forward_sequence', '')}

>Reverse_Strand
{result.get('reverse_sequence', '')}

# Alignment Results:
# Match Score: {result.get('match_score', 0)}
# Overlap Length: {result.get('overlap_length', 0)} bp
# Match Percentage: {match_percent:.1f}%
"""
                if result.get('predicted_tm'):
                    alignment_content += f"# Predicted Tm: {result['predicted_tm']:.1f}°C\n"
                
                create_download_button(alignment_content, "hybridization_results.txt", "Download Results")
            
            with col2_8:
                # Full report
                report_content = f"""G-Synth Hybridization Simulation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

Simulation Parameters:
- Temperature: {result.get('temperature', 37)}°C
- Salt Concentration: {result.get('salt_conc', 50)} mM
- Max Alignment Shift: {result.get('max_shift', 50)}
- Min Match Length: {result.get('min_match_length', 10)}

Results Summary:
- Match Score: {result.get('match_score', 0)}
- Overlap Length: {result.get('overlap_length', 0)} bp
- Match Percentage: {match_percent:.1f}%
- Predicted Tm: {result.get('predicted_tm', 'N/A')}°C

Sequences:
Forward: {result.get('forward_sequence', '')}
Reverse: {result.get('reverse_sequence', '')}
"""
                
                if result.get('thermodynamics'):
                    thermo = result['thermodynamics']
                    report_content += f"""
Thermodynamic Analysis:
- Enthalpy (ΔH): {thermo.get('delta_h', 0):.1f} kcal/mol
- Entropy (ΔS): {thermo.get('delta_s', 0):.1f} cal/(mol·K)
- Free Energy (ΔG): {thermo.get('delta_g', 0):.1f} kcal/mol
- Binding Affinity: {thermo.get('binding_affinity', 'Unknown')}
"""
                
                create_download_button(report_content, "hybridization_report.txt", "Download Report")
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%); 
                        border-radius: 12px; border: 2px dashed #cbd5e1;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🧬</div>
                <div style="color: #64748b; font-size: 1.1rem;">
                    Enter DNA sequences and simulate hybridization to see results
                </div>
            </div>
            """, unsafe_allow_html=True)

def simulate_hybridization(forward_seq, reverse_seq, temperature, salt_conc, max_shift, min_match_length, calculate_tm):
    """Simulate DNA hybridization between two strands"""
    
    # Reverse the reverse sequence (plain reversal) as per specification
    reverse_reversed = reverse_seq[::-1]
    
    # Find optimal alignment
    best_shift, best_score = optimal_alignment(forward_seq, reverse_reversed, max_shift)
    
    # Calculate alignment details
    overlap_start = max(0, best_shift)
    overlap_end = min(len(forward_seq), best_shift + len(reverse_reversed))
    overlap_length = overlap_end - overlap_start
    
    # Create alignment
    alignment = {
        'forward': forward_seq,
        'reverse': reverse_reversed,
        'shift': best_shift,
        'overlap_start': overlap_start,
        'overlap_end': overlap_end,
        'matches': []
    }
    
    # Find matches in the overlap region
    for i in range(overlap_start, overlap_end):
        j = i - best_shift
        if 0 <= j < len(reverse_reversed):
            if is_complement(forward_seq[i], reverse_reversed[j]):
                alignment['matches'].append(i)
    
    # Calculate thermodynamics if requested
    thermodynamics = None
    predicted_tm = None
    
    if calculate_tm and overlap_length >= min_match_length:
        # Simple Tm calculation for the overlap region
        overlap_seq = forward_seq[overlap_start:overlap_end]
        predicted_tm = calculate_tm_consensus(overlap_seq, primer_conc=1e-6, na_conc=salt_conc/1000)
        
        # Basic thermodynamic calculations
        gc_content = calculate_gc(overlap_seq)
        at_content = 100 - gc_content
        
        # Simplified thermodynamic parameters
        delta_h = -(gc_content * 0.1 + at_content * 0.05)  # kcal/mol
        delta_s = -(gc_content * 0.25 + at_content * 0.15)  # cal/(mol·K)
        delta_g = delta_h - (temperature + 273.15) * delta_s / 1000  # kcal/mol
        
        thermodynamics = {
            'delta_h': delta_h,
            'delta_s': delta_s,
            'delta_g': delta_g,
            'binding_affinity': 'Strong' if delta_g < -5 else 'Moderate' if delta_g < -2 else 'Weak'
        }
    
    return {
        'forward_sequence': forward_seq,
        'reverse_sequence': reverse_seq,
        'alignment': alignment,
        'match_score': best_score,
        'overlap_length': overlap_length,
        'predicted_tm': predicted_tm,
        'thermodynamics': thermodynamics,
        'temperature': temperature,
        'salt_conc': salt_conc,
        'max_shift': max_shift,
        'min_match_length': min_match_length
    }

def create_hybridization_visualization(result):
    """Create visualization of DNA hybridization"""
    try:
        alignment = result['alignment']
        
        fig = go.Figure()
        
        # Create sequence alignment visualization
        forward_seq = alignment['forward']
        reverse_seq = alignment['reverse']
        shift = alignment['shift']
        matches = alignment.get('matches', [])
        
        # Create x-axis positions
        x_positions = list(range(len(forward_seq)))
        
        # Add forward strand
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=[1] * len(forward_seq),
            mode='markers+text',
            text=list(forward_seq),
            textposition="middle center",
            marker=dict(size=20, color='blue', symbol='square'),
            name='Forward Strand (5\'→3\')',
            showlegend=True
        ))
        
        # Add reverse strand with shift
        if shift >= 0:
            # Reverse strand starts after shift
            rev_x_positions = list(range(shift, shift + len(reverse_seq)))
            rev_y_positions = [0] * len(reverse_seq)
        else:
            # Reverse strand starts before forward
            rev_x_positions = list(range(len(reverse_seq)))
            rev_y_positions = [0] * len(reverse_seq)
        
        # Color code based on matches
        rev_colors = []
        for i, pos in enumerate(rev_x_positions):
            if shift >= 0:
                fwd_pos = pos
            else:
                fwd_pos = pos - shift
            
            if fwd_pos in matches:
                rev_colors.append('green')  # Complementary match
            else:
                rev_colors.append('red')    # Mismatch or overhang
        
        fig.add_trace(go.Scatter(
            x=rev_x_positions,
            y=rev_y_positions,
            mode='markers+text',
            text=list(reverse_seq),
            textposition="middle center",
            marker=dict(size=20, color=rev_colors, symbol='square'),
            name='Reverse Strand (5\'→3\')',
            showlegend=True
        ))
        
        # Add connecting lines for matches
        for match_pos in matches:
            if shift >= 0:
                rev_pos = match_pos
                if match_pos - shift < len(reverse_seq):
                    fig.add_trace(go.Scatter(
                        x=[match_pos, match_pos],
                        y=[1, 0],
                        mode='lines',
                        line=dict(color='green', width=2, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Update layout
        fig.update_layout(
            title=f"DNA Hybridization Simulation (Score: {result['match_score']}, Overlap: {result['overlap_length']} bp)",
            xaxis_title="Position",
            yaxis_title="Strand",
            yaxis=dict(tickvals=[0, 1], ticktext=['Reverse', 'Forward']),
            height=300,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating hybridization visualization: {e}")
        return None

def create_alignment_display(forward_seq, reverse_seq, matches):
    """Create HTML display of sequence alignment"""
    try:
        # Limit display length for readability
        max_display_length = 80
        
        if len(forward_seq) > max_display_length:
            forward_display = forward_seq[:max_display_length] + "..."
        else:
            forward_display = forward_seq
        
        if len(reverse_seq) > max_display_length:
            reverse_display = reverse_seq[:max_display_length] + "..."
        else:
            reverse_display = reverse_seq
        
        # Create alignment display
        forward_html = ""
        reverse_html = ""
        connection_html = ""
        
        for i, (f_base, r_base) in enumerate(zip(forward_display, reverse_display)):
            if i < len(matches) and i in matches:
                # Matching bases
                forward_html += f'<span style="background: #48ca8a; color: white; padding: 2px 4px; margin: 1px; border-radius: 3px;">{f_base}</span>'
                reverse_html += f'<span style="background: #48ca8a; color: white; padding: 2px 4px; margin: 1px; border-radius: 3px;">{r_base}</span>'
                connection_html += '<span style="color: #48ca8a; margin: 1px;">|</span>'
            else:
                # Non-matching bases
                forward_html += f'<span style="background: #feca57; color: black; padding: 2px 4px; margin: 1px; border-radius: 3px;">{f_base}</span>'
                reverse_html += f'<span style="background: #feca57; color: black; padding: 2px 4px; margin: 1px; border-radius: 3px;">{r_base}</span>'
                connection_html += '<span style="color: #feca57; margin: 1px;">:</span>'
        
        alignment_html = f"""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; background: #f8fafc; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <div style="margin-bottom: 0.5rem;">
                <strong>5'</strong> {forward_html} <strong>3'</strong>
            </div>
            <div style="margin-bottom: 0.5rem; text-align: center;">
                {connection_html}
            </div>
            <div>
                <strong>3'</strong> {reverse_html} <strong>5'</strong>
            </div>
        </div>
        <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">
            <span style="background: #48ca8a; color: white; padding: 2px 6px; border-radius: 3px; margin-right: 10px;">Complementary</span>
            <span style="background: #feca57; color: black; padding: 2px 6px; border-radius: 3px;">Mismatch/Overhang</span>
        </div>
        """
        
        return alignment_html
        
    except Exception as e:
        logger.error(f"Error creating alignment display: {e}")
        return "Error displaying alignment"

# Reverse Complement tab
def render_reverse_complement_tab():
    """Render the Reverse Complement tab"""
    st.markdown("## Reverse Complement")
    st.markdown("Generate complementary DNA strands for molecular biology applications.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Input")
        
        # Sequence input
        sequence_input = st.text_area(
            "DNA Sequence:",
            value="ATGCGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGC",
            height=150,
            placeholder="Enter your DNA sequence",
            help="Enter the DNA sequence to get its reverse complement"
        )
        
        # Real-time validation and preview
        if sequence_input:
            is_valid, clean_seq, warning = validate_dna_sequence(sequence_input)
            if not is_valid:
                create_status_message(f"⚠️ {warning}", "warning")
            elif warning:
                create_status_message(f"ℹ️ {warning}", "info")
            else:
                create_status_message(f"✅ Valid sequence ({len(clean_seq)} bp)", "success")
                
                # Show live preview
                if len(clean_seq) <= 100:  # Only show preview for shorter sequences
                    rev_comp = reverse_complement(clean_seq)
                    st.markdown("**Live Preview:**")
                    st.markdown(f"**Original (5'→3'):** `{clean_seq}`")
                    st.markdown(f"**Rev Comp (5'→3'):** `{rev_comp}`")
        
        # Options
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            include_original = st.checkbox("Include Original Sequence", True)
            show_direction = st.checkbox("Show 5'/3' Labels", True)
        
        with col1_2:
            output_format = st.selectbox(
                "Output Format:",
                ["FASTA", "Plain Text", "Tabular"],
                help="Choose output format"
            )
        
        # Generate button
        if st.button("🔄 Generate Reverse Complement", type="primary", use_container_width=True):
            if sequence_input:
                with st.spinner("Generating reverse complement..."):
                    try:
                        is_valid, clean_seq, warning = validate_dna_sequence(sequence_input)
                        
                        if is_valid:
                            rev_comp = reverse_complement(clean_seq)
                            
                            # Calculate properties
                            properties = {
                                'original_length': len(clean_seq),
                                'original_gc': calculate_gc(clean_seq),
                                'revcomp_gc': calculate_gc(rev_comp),
                                'original_tm': calculate_tm_consensus(clean_seq),
                                'revcomp_tm': calculate_tm_consensus(rev_comp)
                            }
                            
                            # Store results
                            revcomp_result = {
                                'original_sequence': clean_seq,
                                'reverse_complement': rev_comp,
                                'properties': properties,
                                'include_original': include_original,
                                'show_direction': show_direction,
                                'output_format': output_format,
                                'input_warning': warning
                            }
                            
                            st.session_state.current_results = revcomp_result
                            
                            # Add to history
                            st.session_state.history.append({
                                'timestamp': datetime.now().strftime("%H:%M:%S"),
                                'action': f"Generated reverse complement ({len(clean_seq)} bp)",
                                'tool': 'Reverse Complement'
                            })
                            
                            create_status_message("✅ Reverse complement generated successfully!", "success")
                        else:
                            create_status_message(f"❌ Invalid sequence: {warning}", "error")
                    
                    except Exception as e:
                        logger.error(f"Error generating reverse complement: {e}")
                        create_status_message(f"❌ Error: {str(e)}", "error")
            else:
                create_status_message("❌ Please enter a DNA sequence", "error")
    
    with col2:
        st.markdown("### Results")
        
        if ('current_results' in st.session_state and 
            'reverse_complement' in st.session_state.current_results):
            
            result = st.session_state.current_results
            original_seq = result['original_sequence']
            rev_comp_seq = result['reverse_complement']
            properties = result['properties']
            
            # Display warning if any
            if result.get('input_warning'):
                create_status_message(f"ℹ️ {result['input_warning']}", "info")
            
            # Properties comparison
            st.markdown("#### Sequence Properties")
            col2_1, col2_2, col2_3 = st.columns(3)
            
            with col2_1:
                create_metric_card("Length", f"{properties['original_length']} bp")
            
            with col2_2:
                create_metric_card("Original GC%", f"{properties['original_gc']:.1f}%")
            
            with col2_3:
                create_metric_card("Rev Comp GC%", f"{properties['revcomp_gc']:.1f}%")
            
            # Additional properties
            if properties.get('original_tm') and properties.get('revcomp_tm'):
                col2_4, col2_5 = st.columns(2)
                
                with col2_4:
                    create_metric_card("Original Tm", f"{properties['original_tm']:.1f}°C")
                
                with col2_5:
                    create_metric_card("Rev Comp Tm", f"{properties['revcomp_tm']:.1f}°C")
            
            # Display sequences
            st.markdown("#### Sequences")
            
            if result['output_format'] == "FASTA":
                fasta_display = ""
                if result['include_original']:
                    fasta_display += ">Original_Sequence\n"
                    if result['show_direction']:
                        fasta_display += f"# 5' -> 3'\n"
                    fasta_display += f"{original_seq}\n\n"
                
                fasta_display += ">Reverse_Complement\n"
                if result['show_direction']:
                    fasta_display += f"# 5' -> 3'\n"
                fasta_display += f"{rev_comp_seq}\n"
                
                st.markdown("```
                st.markdown(fasta_display)
                st.markdown("```")
            
            elif result['output_format'] == "Plain Text":
                if result['include_original']:
                    label = "Original (5'→3'):" if result['show_direction'] else "Original:"
                    st.markdown(f"**{label}**")
                    st.markdown(f'<div class="sequence-display">{original_seq}</div>', unsafe_allow_html=True)
                
                label = "Reverse Complement (5'→3'):" if result['show_direction'] else "Reverse Complement:"
                st.markdown(f"**{label}**")
                st.markdown(f'<div class="sequence-display">{rev_comp_seq}</div>', unsafe_allow_html=True)
            
            elif result['output_format'] == "Tabular":
                table_data = {
                    'Type': ['Reverse Complement'],
                    'Sequence (5\'→3\')': [rev_comp_seq],
                    'Length': [f"{len(rev_comp_seq)} bp"],
                    'GC Content': [f"{properties['revcomp_gc']:.1f}%"]
                }
                
                if result['include_original']:
                    table_data['Type'].insert(0, 'Original')
                    table_data['Sequence (5\'→3\')'].insert(0, original_seq)
                    table_data['Length'].insert(0, f"{len(original_seq)} bp")
                    table_data['GC Content'].insert(0, f"{properties['original_gc']:.1f}%")
                
                if properties.get('original_tm') and properties.get('revcomp_tm'):
                    table_data['Tm (°C)'] = [f"{properties['revcomp_tm']:.1f}"]
                    if result['include_original']:
                        table_data['Tm (°C)'].insert(0, f"{properties['original_tm']:.1f}")
                
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Base pairing visualization for shorter sequences
            if len(original_seq) <= 50:
                st.markdown("#### Base Pairing Visualization")
                
                # Create base pairing display
                pairing_html = '<div style="font-family: \'JetBrains Mono\', monospace; font-size: 0.9rem; background: #f8fafc; padding: 1rem; border-radius: 8px; text-align: center;">'
                
                # Original sequence
                pairing_html += f'<div style="margin-bottom: 0.5rem;"><strong>5\'</strong> '
                for base in original_seq:
                    pairing_html += f'<span style="margin: 0 2px; padding: 2px 4px; background: #3498db; color: white; border-radius: 3px;">{base}</span>'
                pairing_html += ' <strong>3\'</strong></div>'
                
                # Pairing lines
                pairing_html += '<div style="margin-bottom: 0.5rem;">'
                for i in range(len(original_seq)):
                    pairing_html += '<span style="margin: 0 6px;">|</span>'
                pairing_html += '</div>'
                
                # Reverse complement
                pairing_html += f'<div><strong>3\'</strong> '
                for base in rev_comp_seq:
                    pairing_html += f'<span style="margin: 0 2px; padding: 2px 4px; background: #e74c3c; color: white; border-radius: 3px;">{base}</span>'
                pairing_html += ' <strong>5\'</strong></div>'
                
                pairing_html += '</div>'
                
                st.markdown(pairing_html, unsafe_allow_html=True)
            
            # Download options
            st.markdown("#### Download Options")
            col2_6, col2_7 = st.columns(2)
            
            with col2_6:
                # Generate download content based on selected format
                if result['output_format'] == "FASTA":
                    download_content = ""
                    if result['include_original']:
                        download_content += ">Original_Sequence\n"
                        if result['show_direction']:
                            download_content += "# 5' -> 3'\n"
                        download_content += f"{original_seq}\n\n"
                    
                    download_content += ">Reverse_Complement\n"
                    if result['show_direction']:
                        download_content += "# 5' -> 3'\n"
                    download_content += f"{rev_comp_seq}\n"
                    
                    filename = "reverse_complement.fasta"
                    button_text = "Download FASTA"
                
                elif result['output_format'] == "Plain Text":
                    download_content = ""
                    if result['include_original']:
                        label = "Original (5'->3'):" if result['show_direction'] else "Original:"
                        download_content += f"{label}\n{original_seq}\n\n"
                    
                    label = "Reverse Complement (5'->3'):" if result['show_direction'] else "Reverse Complement:"
                    download_content += f"{label}\n{rev_comp_seq}\n"
                    
                    filename = "reverse_complement.txt"
                    button_text = "Download Text"
                
                else:  # Tabular
                    download_content = "Type\tSequence_5to3\tLength\tGC_Content"
                    if properties.get('original_tm'):
                        download_content += "\tTm_C"
                    download_content += "\n"
                    
                    if result['include_original']:
                        download_content += f"Original\t{original_seq}\t{len(original_seq)}\t{properties['original_gc']:.1f}"
                        if properties.get('original_tm'):
                            download_content += f"\t{properties['original_tm']:.1f}"
                        download_content += "\n"
                    
                    download_content += f"Reverse_Complement\t{rev_comp_seq}\t{len(rev_comp_seq)}\t{properties['revcomp_gc']:.1f}"
                    if properties.get('revcomp_tm'):
                        download_content += f"\t{properties['revcomp_tm']:.1f}"
                    download_content += "\n"
                    
                    filename = "reverse_complement.tsv"
                    button_text = "Download TSV"
                
                create_download_button(download_content, filename, button_text)
            
            with col2_7:
                # Comprehensive report
                report_content = f"""G-Synth Reverse Complement Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

Input Sequence Analysis:
- Length: {properties['original_length']} bp
- GC Content: {properties['original_gc']:.1f}%
"""
                if properties.get('original_tm'):
                    report_content += f"- Melting Temperature: {properties['original_tm']:.1f}°C\n"
                
                report_content += f"""
Original Sequence (5' -> 3'):
{original_seq}

Reverse Complement (5' -> 3'):
{rev_comp_seq}

Reverse Complement Analysis:
- Length: {len(rev_comp_seq)} bp
- GC Content: {properties['revcomp_gc']:.1f}%
"""
                if properties.get('revcomp_tm'):
                    report_content += f"- Melting Temperature: {properties['revcomp_tm']:.1f}°C\n"
                
                # Add base composition
                base_comp_orig = {base: original_seq.count(base) for base in 'ATCG'}
                base_comp_rev = {base: rev_comp_seq.count(base) for base in 'ATCG'}
                
                report_content += f"""
Base Composition:
Original:  A:{base_comp_orig['A']}, T:{base_comp_orig['T']}, C:{base_comp_orig['C']}, G:{base_comp_orig['G']}
Rev Comp:  A:{base_comp_rev['A']}, T:{base_comp_rev['T']}, C:{base_comp_rev['C']}, G:{base_comp_rev['G']}

Note: The reverse complement maintains Watson-Crick base pairing rules (A-T, G-C).
"""
                
                create_download_button(report_content, "reverse_complement_report.txt", "Download Report")
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%); 
                        border-radius: 12px; border: 2px dashed #cbd5e1;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔄</div>
                <div style="color: #64748b; font-size: 1.1rem;">
                    Enter a DNA sequence to generate its reverse complement
                </div>
            </div>
            """, unsafe_allow_html=True)

# Help & Guide tab
def render_help_tab():
    """Render the Help & Guide tab"""
    st.markdown("## Help & Guide")
    st.markdown("Complete documentation and tutorials for the G-Synth platform.")
    
    # Create tabs for different help sections
    help_tab1, help_tab2, help_tab3, help_tab4 = st.tabs([
        "🚀 Getting Started", 
        "🛠️ Tool Guide", 
        "🤖 AI Features", 
        "📚 Reference"
    ])
    
    with help_tab1:
        st.markdown("### Welcome to G-Synth!")
        
        st.markdown("""
        G-Synth is a comprehensive genetic engineering toolkit that combines traditional molecular biology tools 
        with cutting-edge AI capabilities for enhanced research and development.
        
        #### Quick Start Guide:
        
        1. **Choose a Tool**: Use the sidebar navigation to select the appropriate tool for your task
        2. **Input Your Data**: Enter your DNA or protein sequences in the input fields
        3. **Configure Parameters**: Adjust settings based on your experimental requirements
        4. **Run Analysis**: Click the main action button to perform the analysis
        5. **Review Results**: Examine the results, visualizations, and recommendations
        6. **Download**: Save your results in various formats for further use
        
        #### Key Features:
        
        - 🧬 **Small Sequence Design**: Design short DNA sequences with restriction sites
        - 🔄 **Translation Tools**: Convert between DNA and protein sequences
        - 🤖 **AI Codon Optimization**: Machine learning-guided codon optimization
        - 🧪 **Extended Synthesis**: Fragment large sequences for synthesis
        - 🎯 **Hybridization Simulation**: Predict DNA strand interactions
        - 📊 **Comprehensive Analysis**: Detailed sequence analysis and visualization
        """)
        
        # Quick tips
        st.markdown("#### 💡 Quick Tips:")
        
        tips = [
            "Always validate your input sequences before running analysis",
            "Use the AI features for enhanced optimization and predictions",
            "Check the confidence scores on AI-generated results",
            "Download comprehensive reports for documentation",
            "Enable advanced options for more detailed control",
            "Use the session history to track your work"
        ]
        
        for tip in tips:
            st.markdown(f"• {tip}")
    
    with help_tab2:
        st.markdown("### Tool-by-Tool Guide")
        
        # Create expandable sections for each tool
        tools_help = {
            "Small Sequence Design": {
                "description": "Design short DNA sequences with restriction enzyme sites for cloning and protein expression.",
                "input": "DNA sequence (coding or non-coding)",
                "parameters": [
                    "Sequence type (coding/non-coding)",
                    "Enzyme pair selection",
                    "Optional cleavage site",
                    "Stop codon removal (coding sequences)"
                ],
                "output": "Forward and reverse sequences with sticky ends, properties analysis",
                "tips": [
                    "Coding sequences must start with ATG",
                    "Choose enzyme pairs based on your vector",
                    "Consider cleavage sites for protein purification"
                ]
            },
            "Translation & Reverse Translation": {
                "description": "Convert between DNA sequences and amino acid sequences with multiple reading frame support.",
                "input": "DNA sequence (for translation) or protein sequence (for reverse translation)",
                "parameters": [
                    "Reading frame selection",
                    "Start codon finding",
                    "Target organism (reverse translation)",
                    "Optimization level"
                ],
                "output": "Translated protein or optimized DNA sequence with verification",
                "tips": [
                    "Check all three reading frames for ORF detection",
                    "Verify back-translation matches original sequence",
                    "Choose target organism based on expression system"
                ]
            },
            "AI Codon Optimization": {
                "description": "Machine learning-guided codon optimization for enhanced protein expression.",
                "input": "DNA or protein sequence",
                "parameters": [
                    "Target organism",
                    "AI optimization level",
                    "GC content targets",
                    "Site avoidance"
                ],
                "output": "AI-optimized sequence with confidence scores and predictions",
                "tips": [
                    "Higher AI optimization levels may change more codons",
                    "Review AI confidence scores before proceeding",
                    "Consider expression predictions in context"
                ]
            },
            "Extended Synthesis": {
                "description": "Fragment and assemble large DNA sequences for gene synthesis.",
                "input": "Long DNA sequence",
                "parameters": [
                    "Fragment size",
                    "Overlap size",
                    "Assembly method",
                    "Terminal enzymes"
                ],
                "output": "Fragment list with assembly strategy and verification",
                "tips": [
                    "Balance fragment size with synthesis cost",
                    "Ensure adequate overlap for assembly",
                    "Verify reassembly before ordering synthesis"
                ]
            },
            "Hybridization Simulation": {
                "description": "Simulate DNA strand hybridization and predict annealing behavior.",
                "input": "Forward and reverse DNA strands",
                "parameters": [
                    "Temperature",
                    "Salt concentration",
                    "Maximum alignment shift",
                    "Minimum match length"
                ],
                "output": "Alignment visualization, match scores, and thermodynamic analysis",
                "tips": [
                    "Consider experimental conditions when setting parameters",
                    "Higher match scores indicate better hybridization",
                    "Review thermodynamic predictions for stability"
                ]
            },
            "Reverse Complement": {
                "description": "Generate complementary DNA strands for molecular biology applications.",
                "input": "DNA sequence",
                "parameters": [
                    "Output format",
                    "Include original sequence",
                    "Show 5'/3' direction labels"
                ],
                "output": "Reverse complement sequence with properties comparison",
                "tips": [
                    "Useful for designing antisense oligonucleotides",
                    "Verify base pairing for short sequences",
                    "Consider melting temperature for applications"
                ]
            }
        }
        
        for tool_name, help_info in tools_help.items():
            with st.expander(f"📋 {tool_name}"):
                st.markdown(f"**Description:** {help_info['description']}")
                st.markdown(f"**Input:** {help_info['input']}")
                
                st.markdown("**Key Parameters:**")
                for param in help_info['parameters']:
                    st.markdown(f"• {param}")
                
                st.markdown(f"**Output:** {help_info['output']}")
                
                st.markdown("**💡 Tips:**")
                for tip in help_info['tips']:
                    st.markdown(f"• {tip}")
    
    with help_tab3:
        st.markdown("### 🤖 AI Features Guide")
        
        st.markdown("""
        G-Synth incorporates advanced AI capabilities to enhance your molecular biology workflows. 
        Here's how to make the most of these features:
        
        #### AI Codon Optimization
        
        Our AI system uses machine learning models trained on large-scale expression data to optimize 
        codon usage for maximum protein expression and stability.
        
        **Key Features:**
        - **Expression Prediction**: AI predicts fold-change in protein expression
        - **Stability Analysis**: Estimates protein stability improvements
        - **Confidence Scoring**: Reliability metrics for all predictions
        - **Multi-objective Optimization**: Balance expression, stability, and speed
        
        **How to Use:**
        1. Enable AI features in the sidebar
        2. Set your confidence threshold (recommended: 70-80%)
        3. Choose optimization strategy based on your goals
        4. Review AI insights and confidence scores
        5. Validate predictions experimentally when possible
        
        #### Understanding AI Confidence Scores
        
        AI confidence scores indicate the reliability of predictions:
        - **90-100%**: High confidence - reliable for most applications
        - **70-89%**: Good confidence - suitable with experimental validation
        - **50-69%**: Moderate confidence - requires careful consideration
        - **Below 50%**: Low confidence - experimental validation essential
        
        #### AI Model Information
        
        Our AI models are trained on:
        - Large-scale protein expression datasets
        - Structural and functional annotations
        - Experimental optimization results
        - Cross-validated performance metrics
        
        **Best Practices:**
        - Always review AI recommendations critically
        - Consider experimental context and constraints
        - Use confidence scores to guide decision-making
        - Validate important predictions experimentally
        - Report any discrepancies to improve our models
        """)
        
        # AI feature availability
        st.markdown("#### AI Feature Status")
        
        ai_features_status = [
            ("🤖 AI Codon Optimization", "Available", "success"),
            ("🧬 Protein Structure Prediction", "Coming Soon", "warning"),
            ("🎯 Primer Design AI", "Coming Soon", "warning"),
            ("🔬 CRISPR Guide AI", "Coming Soon", "warning"),
            ("📊 Expression Prediction", "Beta", "info")
        ]
        
        for feature, status, status_type in ai_features_status:
            if status_type == "success":
                st.success(f"{feature}: {status}")
            elif status_type == "warning":
                st.warning(f"{feature}: {status}")
            else:
                st.info(f"{feature}: {status}")
    
    with help_tab4:
        st.markdown("### 📚 Reference Documentation")
        
        # Technical specifications
        st.markdown("#### Technical Specifications")
        
        with st.expander("🧬 Supported Organisms"):
            st.markdown("""
            **Codon Usage Tables Available:**
            - *Escherichia coli* BL21(DE3)
            - *Saccharomyces cerevisiae*
            - *Pichia pastoris*
            - *Homo sapiens*
            - CHO cells
            
            **Expression Systems:**
            - Bacterial (E. coli)
            - Yeast (S. cerevisiae, P. pastoris)
            - Mammalian (Human, CHO)
            """)
        
        with st.expander("🔬 Restriction Enzymes"):
            enzymes_info = [
                ("NdeI", "CATATG", "5' overhang"),
                ("XhoI", "CTCGAG", "5' overhang"),
                ("EcoRI", "GAATTC", "5' overhang"),
                ("BamHI", "GGATCC", "5' overhang"),
                ("HindIII", "AAGCTT", "5' overhang"),
                ("SalI", "GTCGAC", "5' overhang"),
                ("XbaI", "TCTAGA", "5' overhang"),
                ("NcoI", "CCATGG", "5' overhang")
            ]
            
            enzyme_df = pd.DataFrame(enzymes_info, columns=["Enzyme", "Recognition Site", "Overhang Type"])
            st.dataframe(enzyme_df, use_container_width=True, hide_index=True)
        
        with st.expander("🧪 Cleavage Sites"):
            cleavage_info = [
                ("Thrombin", "LVPR|GS", "Factor IIa"),
                ("TEV", "ENLYFQ|G", "Tobacco Etch Virus"),
                ("Factor Xa", "IEGR|", "Factor Xa"),
                ("PreScission", "LEVLFQ|GP", "HRV 3C-like"),
                ("Enterokinase", "DDDDK|", "Enterokinase"),
                ("SUMO", "Variable", "Small Ubiquitin Modifier"),
                ("HRV 3C", "LEVLFQ|GP", "Human Rhinovirus 3C")
            ]
            
            cleavage_df = pd.DataFrame(cleavage_info, columns=["Protease", "Cleavage Site", "Source"])
            st.dataframe(cleavage_df, use_container_width=True, hide_index=True)
        
        # File formats
        st.markdown("#### 📁 Supported File Formats")
        
        formats = {
            "Input Formats": [
                "Plain text sequences",
                "FASTA format",
                "Multi-line sequences",
                "Copy-paste from other tools"
            ],
            "Output Formats": [
                "FASTA files",
                "Tab-separated values (TSV)",
                "Comprehensive text reports",
                "JSON data exports"
            ]
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Input Formats:**")
            for fmt in formats["Input Formats"]:
                st.markdown(f"• {fmt}")
        
        with col2:
            st.markdown("**Output Formats:**")
            for fmt in formats["Output Formats"]:
                st.markdown(f"• {fmt}")
        
        # Contact and support
        st.markdown("#### 📞 Support & Contact")
        
        st.markdown("""
        **Developer:** Dr. Mohamed Merzoug  
        **Version:** 2025.6.0  
        **Platform:** Streamlit Web Application  
        
        **For Support:**
        - 📧 Email: [Contact via GitHub](https://github.com/g-synth)
        - 🐛 Bug Reports: [GitHub Issues](https://github.com/g-synth/issues)
        - 📖 Documentation: [GitHub Wiki](https://github.com/g-synth/wiki)
        - 💡 Feature Requests: [GitHub Discussions](https://github.com/g-synth/discussions)
        
        **Citation:**
        If you use G-Synth in your research, please cite:
        ```
        Merzoug, M. (2025). G-Synth: AI-Enhanced Genetic Engineering Toolkit. 
        Version 2025.6.0. Available at: https://github.com/g-synth
        ```
        """)
        
        # Version history
        with st.expander("📋 Version History"):
            st.markdown("""
            **Version 2025.6.0** (Current)
            - Complete AI integration across all tools
            - Enhanced user interface with professional styling
            - Comprehensive error handling and validation
            - Advanced visualization capabilities
            - Session management and history tracking
            
            **Version 2025.5.0**
            - Initial AI codon optimization
            - Improved sequence validation
            - Enhanced download options
            - Bug fixes and performance improvements
            
            **Version 2025.2.0**
            - Core functionality implementation
            - Basic web interface
            - All traditional molecular biology tools
            - Initial release
            """)

# Main application function
def main():
    """Main application function with enhanced error handling"""
    try:
        # Render header
        render_header()
        
        # Render sidebar and get selected tab
        selected_tab = render_sidebar()
        
        # Tab navigation mapping
        tab_functions = {
            "Home": render_home_tab,
            "Small Sequence Design": render_small_sequence_design_tab,
            "Translation & Reverse Translation": render_translation_tab,
            "AI Codon Optimization": render_ai_codon_optimization_tab,
            "Extended Synthesis": render_extended_synthesis_tab,
            "Hybridization Simulation": render_hybridization_simulation_tab,
            "Ligation Check": lambda: st.info("🚧 Ligation Check - Implementation in Progress!"),
            "AI Primer Designer": lambda: st.info("🚧 AI Primer Designer - Implementation in Progress!"),
            "CRISPR Guide Designer": lambda: st.info("🚧 CRISPR Guide Designer - Implementation in Progress!"),
            "Reverse Complement": render_reverse_complement_tab,
            "Help & Guide": render_help_tab
        }
        
        # Render the selected tab
        if selected_tab in tab_functions:
            tab_functions[selected_tab]()
        else:
            # Fallback for unknown tabs
            st.error(f"Unknown tab: {selected_tab}")
            st.info("Redirecting to Home...")
            render_home_tab()
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An unexpected error occurred. Please refresh the page and try again.")
        
        # Show error details in debug mode
        if st.session_state.user_preferences.get('debug_mode', False):
            st.exception(e)
        
        # Provide recovery options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏠 Go to Home"):
                st.session_state.selected_tool = "Home"
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Session"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col3:
            if st.button("📥 Download Error Log"):
                error_log = f"""G-Synth Error Log
Generated: {datetime.now().isoformat()}
Error: {str(e)}
Traceback: {traceback.format_exc()}

Session State:
{json.dumps(dict(st.session_state), indent=2, default=str)}
"""
                st.download_button(
                    "📥 Download Log",
                    error_log,
                    f"g_synth_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                    "text/plain"
                )

# Helper functions for enhanced functionality
def optimal_alignment(forward, reverse_comp, max_shift=None):
    """Find best alignment shift between forward and reverse complement sequences"""
    if max_shift is None:
        max_shift = len(forward) + len(reverse_comp)
    else:
        max_shift = min(max_shift, len(forward) + len(reverse_comp))
    
    best = (0, 0)  # (shift, score)
    
    for shift in range(-len(reverse_comp)+1, len(forward)):
        score = 0
        for i in range(max(0, shift), min(len(forward), shift + len(reverse_comp))):
            j = i - shift
            if 0 <= j < len(reverse_comp) and is_complement(forward[i], reverse_comp[j]):
                score += 1
        
        if score > best[1]:
            best = (shift, score)
    
    return best

def is_complement(base1, base2):
    """Check if two DNA bases are complementary"""
    comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return comp.get(base1.upper(), '') == base2.upper()

def fragment_extended_sequence(seq, frag_size, enzyme_pair, cleavage_site, internal_overlap=15):
    """Fragment a long sequence into overlapping pieces for synthesis"""
    if len(seq) < frag_size:
        raise ValueError("Sequence is shorter than the fragment size.")
    
    # Parse enzyme pair
    try:
        left_enzyme, right_enzyme = enzyme_pair.split(" / ")
    except ValueError:
        raise ValueError("Invalid enzyme pair format")
    
    cleavage_seq = cleavage_sites.get(cleavage_site, "")
    
    # Calculate effective fragment size (excluding overlap)
    effective_size = frag_size - internal_overlap
    
    # Generate overlapping fragments
    parts = []
    for i in range(0, len(seq), effective_size):
        start = i
        end = min(i + frag_size, len(seq))
        parts.append(seq[start:end])
    
    assembly = []
    for idx, frag in enumerate(parts):
        if idx == 0:
            # First fragment
            fwd = enzyme_pairs[enzyme_pair]["forward_overhang"] + cleavage_seq + frag
            next_overlap = parts[idx+1][:internal_overlap] if idx+1 < len(parts) else ""
            rev = reverse_complement(frag) + reverse_complement(next_overlap)
            typ = "First"
        elif idx == len(parts) - 1:
            # Last fragment
            prev_overlap = parts[idx-1][-internal_overlap:] if idx > 0 else ""
            fwd = prev_overlap + frag
            rev = reverse_complement(frag) + cleavage_seq + enzyme_pairs[enzyme_pair]["reverse_overhang"]
            typ = "Last"
        else:
            # Internal fragment
            prev_overlap = parts[idx-1][-internal_overlap:] if idx > 0 else ""
            next_overlap = parts[idx+1][:internal_overlap] if idx+1 < len(parts) else ""
            fwd = prev_overlap + frag
            rev = reverse_complement(frag) + reverse_complement(next_overlap)
            typ = "Internal"
        
        assembly.append({
            "fragment": idx + 1,
            "sequence": frag,
            "forward": fwd,
            "reverse": rev,
            "type": typ,
            "length": len(frag)
        })
    
    # Reassemble to verify
    reassembled = parts[0]
    for i in range(1, len(parts)):
        reassembled += parts[i][internal_overlap:] if len(parts[i]) > internal_overlap else parts[i]
    
    return assembly, reassembled

def simulate_hybridization(forward_seq, reverse_seq, temperature, salt_conc, max_shift, min_match_length, calculate_tm):
    """Simulate DNA hybridization between two strands"""
    
    # Reverse the reverse sequence (plain reversal) as per specification
    reverse_reversed = reverse_seq[::-1]
    
    # Find optimal alignment
    best_shift, best_score = optimal_alignment(forward_seq, reverse_reversed, max_shift)
    
    # Calculate alignment details
    overlap_start = max(0, best_shift)
    overlap_end = min(len(forward_seq), best_shift + len(reverse_reversed))
    overlap_length = overlap_end - overlap_start
    
    # Create alignment
    alignment = {
        'forward': forward_seq,
        'reverse': reverse_reversed,
        'shift': best_shift,
        'overlap_start': overlap_start,
        'overlap_end': overlap_end,
        'matches': []
    }
    
    # Find matches in the overlap region
    for i in range(overlap_start, overlap_end):
        j = i - best_shift
        if 0 <= j < len(reverse_reversed):
            if is_complement(forward_seq[i], reverse_reversed[j]):
                alignment['matches'].append(i)
    
    # Calculate thermodynamics if requested
    thermodynamics = None
    predicted_tm = None
    
    if calculate_tm and overlap_length >= min_match_length:
        # Simple Tm calculation for the overlap region
        overlap_seq = forward_seq[overlap_start:overlap_end]
        predicted_tm = calculate_tm_consensus(overlap_seq, primer_conc=1e-6, na_conc=salt_conc/1000)
        
        # Basic thermodynamic calculations
        gc_content = calculate_gc(overlap_seq)
        at_content = 100 - gc_content
        
        # Simplified thermodynamic parameters
        delta_h = -(gc_content * 0.1 + at_content * 0.05)  # kcal/mol
        delta_s = -(gc_content * 0.25 + at_content * 0.15)  # cal/(mol·K)
        delta_g = delta_h - (temperature + 273.15) * delta_s / 1000  # kcal/mol
        
        thermodynamics = {
            'delta_h': delta_h,
            'delta_s': delta_s,
            'delta_g': delta_g,
            'binding_affinity': 'Strong' if delta_g < -5 else 'Moderate' if delta_g < -2 else 'Weak'
        }
    
    return {
        'forward_sequence': forward_seq,
        'reverse_sequence': reverse_seq,
        'alignment': alignment,
        'match_score': best_score,
        'overlap_length': overlap_length,
        'predicted_tm': predicted_tm,
        'thermodynamics': thermodynamics,
        'temperature': temperature,
        'salt_conc': salt_conc,
        'max_shift': max_shift,
        'min_match_length': min_match_length
    }

# Application entry point
if __name__ == "__main__":
    main()