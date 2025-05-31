#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G-Synth: Genetic Engineering Toolkit - Streamlit Version
Version 2025.5.0 - Enhanced Interactive Web Interface

A comprehensive toolkit for gene synthesis and molecular cloning.
Developed by Dr. Mohamed Merzoug

This comprehensive version includes all original features with enhanced UI:
- Small Sequence Design
- Translation & Reverse Translation
- Codon Optimization  
- Extended Synthesis (Long Sequences)
- Hybridization Simulation
- Ligation Check
- Primer Generator
- Reverse Complement
- Help & Guide
"""

import streamlit as st
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
from typing import Dict, List, Tuple, Optional, Union
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing optional dependencies
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

# Page configuration with enhanced styling
st.set_page_config(
    page_title="G-Synth: Genetic Engineering Toolkit",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/g-synth/help',
        'Report a bug': "https://github.com/g-synth/issues",
        'About': "G-Synth v2025.5.0 - Advanced Genetic Engineering Toolkit by Dr. Mohamed Merzoug"
    }
)

# Enhanced CSS for professional styling matching RennetOptiMax Pro
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    .main {
        padding-top: 1rem;
    }
    
    /* Custom Header */
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
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Professional Cards */
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
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .card-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .card-description {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: #64748b;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border-radius: 8px;
        border: none;
        padding: 0.75rem 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(102, 126, 234, 0.25);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px -3px rgba(102, 126, 234, 0.3);
        background: linear-gradient(135deg, #5a67d8 0%, #6b5b95 100%);
    }
    
    /* Sequence Display */
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
    
    /* Biological Feature Highlighting */
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
    
    .linker { 
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        color: white;
        padding: 2px 4px;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .cleavage-site { 
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        color: white;
        padding: 2px 4px;
        border-radius: 4px;
        font-weight: 500;
    }
    
    /* Results Panel */
    .results-panel {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Metrics */
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
    
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Success/Warning/Error Messages */
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
    
    /* Enhanced Tables */
    .dataframe {
        border: none !important;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .dataframe thead th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 1rem;
        border: none;
    }
    
    .dataframe tbody td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    /* Progress Bars */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 4px;
    }
    
    /* Navigation Pills */
    .nav-pill {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        margin: 0.25rem;
        display: inline-block;
    }
    
    /* Tool Status Indicators */
    .status-ready { color: #10b981; font-weight: 600; }
    .status-processing { color: #f59e0b; font-weight: 600; }
    .status-complete { color: #667eea; font-weight: 600; }
    .status-error { color: #ef4444; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Constants from original application (preserved exactly)
NUCLEOTIDES = ['A', 'T', 'C', 'G']
AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '*']

# Genetic code table (from original)
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

# Enzyme pairs (from original)
enzyme_pairs = {
    "NdeI / XhoI": {"forward_overhang": "TA", "reverse_overhang": "TCGA"},
    "NdeI / EcoRI": {"forward_overhang": "TA", "reverse_overhang": "AATT"},
    "BamHI / EcoRI": {"forward_overhang": "GATC", "reverse_overhang": "AATT"},
    "BamHI / XhoI": {"forward_overhang": "GATC", "reverse_overhang": "TCGA"},
    "SalI / XbaI": {"forward_overhang": "TCGAC", "reverse_overhang": "TCTAG"}
}

# Enzyme linkers (from original)
enzyme_linkers = {
    "NdeI": "CATATG", "XhoI": "CTCGAG", "EcoRI": "GAATTC", "BamHI": "GGATCC",
    "HindIII": "AAGCTT", "SalI": "GTCGAC", "XbaI": "TCTAGA", "NcoI": "CCATGG",
    "KpnI": "GGTACC", "SacI": "GAGCTC", "NotI": "GCGGCCGC", "SpeI": "ACTAGT",
    "PstI": "CTGCAG", "BglII": "AGATCT"
}

# Cleavage sites (from original)
cleavage_sites = {
    "Thrombin": "CTGGTGCCGCGTGGTTCT",
    "TEV": "GAAAACCTGTATTTTCAGGGC",
    "Factor Xa": "ATCGAAGGTCGT",
    "PreScission": "CTGGAAGTGCTGTTCCAGGGCCCA",
    "Enterokinase": "GATGACGATGACAAG",
    "SUMO": "CTGCAGGACTCAGAGG",
    "HRV 3C": "CTGGAAGTTCTGTTCCAGGGGCCC"
}

# Fixed elements (from original)
FIXED_ELEMENTS = {
    "left_linker": "TGGGTTCTTCT",
    "his_tag": "CACCACCACCACCACCAC",
    "right_linker": "ACTCTTCTGGT"
}

# Codon usage tables (from original)
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

# Codon frequency tables (from original)
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

# SSD specific constants (from original)
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

SSD_CLEAVAGE_SITES = {
    "Thrombin": "CTGGTGCCGCGTGGTTCT",
    "TEV": "GAAAACCTGTATTTTCAGGGC",
    "Factor Xa": "ATCGAGGGAAGG",
    "PreScission": "CTGGAAGTGCTGTTCCAGGGCCCA",
    "Enterokinase": "GATGACGATGACAAG",
    "SUMO": "CTGCAGGACTCAGAGG",
    "HRV 3C": "CTGGAAGTTCTGTTCCAGGGGCCC"
}

# NN Tm calculation parameters (from original)
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

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = {}
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'theme': 'light',
        'default_organism': 'E. coli BL21',
        'show_advanced_options': False
    }
if 'vector_digested' not in st.session_state:
    st.session_state.vector_digested = False
if 'digested_vector' not in st.session_state:
    st.session_state.digested_vector = None

# Core biological functions (preserved from original)

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

def reverse_translate_to_dna(prot, target_organism="E. coli BL21"):
    """Convert a protein sequence to DNA using optimal codons for the specified organism."""
    if target_organism not in CODON_USAGE_TABLES:
        target_organism = "E. coli BL21"
    
    codon_table = CODON_USAGE_TABLES[target_organism]
    return "".join(codon_table.get(aa, ["NNN"])[0] for aa in prot if aa in codon_table)

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
    
    # Breslauer parameters
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
    
    # SantaLucia parameters
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
    
    # Sugimoto parameters
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
    
    R = 1.987  # cal/(mol*K)
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
        
        delta_s = delta_s + (-10.8)  # Entropy correction
        c = primer_conc / 4
        tm_kelvin = (delta_h * 1000) / (delta_s + R * math.log(c))
        salt_correction = 16.6 * math.log10(na_conc)
        tm_celsius = tm_kelvin - 273.15 + salt_correction
        tm_values.append(tm_celsius)
    
    return round(sum(tm_values) / len(tm_values), 1)

def advanced_codon_optimization(sequence, target_organism="E. coli BL21", optimization_parameters=None, is_protein=False):
    """Perform advanced codon optimization on a DNA or protein sequence."""
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
                
                # Apply GC content constraint
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
                
                # Avoid restriction sites
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
                
                optimized += best_codon
                if best_codon != codon:
                    codon_changes += 1
            else:
                optimized += codon
        
        results["optimized_sequence"] = optimized
        results["codon_changes"] = codon_changes
        
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

def find_orfs(seq):
    """Find all open reading frames in a DNA sequence."""
    seq = clean_dna_sequence(seq)
    orfs = []
    
    for frame in range(3):
        i = frame
        while i < len(seq) - 2:
            if seq[i:i+3] == "ATG":
                start = i
                for j in range(i+3, len(seq) - 2, 3):
                    if seq[j:j+3] in ["TAA", "TAG", "TGA"]:
                        orfs.append((start, j+3, frame))
                        i = j + 3
                        break
                else:
                    i += 3
            else:
                i += 1
    
    return orfs

def design_cloning_primers(forward_seq, reverse_seq, fwd_enzyme, rev_enzyme, primer_conc=500, custom_prefix="TGCATC"):
    """Design primers for molecular cloning with restriction enzyme sites."""
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

def optimal_alignment(forward, reverse_comp, max_shift=None):
    """Find best alignment shift between forward and reverse complement."""
    if max_shift is None:
        max_shift = len(forward) + len(reverse_comp)
    else:
        max_shift = min(max_shift, len(forward) + len(reverse_comp))
    
    best = (0, 0)  # (shift, score)
    
    def is_complement(base1, base2):
        comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        return comp.get(base1.upper(), '') == base2.upper()
    
    for shift in range(-len(reverse_comp)+1, len(forward)):
        score = 0
        for i in range(max(0, shift), min(len(forward), shift + len(reverse_comp))):
            j = i - shift
            if 0 <= j < len(reverse_comp) and is_complement(forward[i], reverse_comp[j]):
                score += 1
        
        if score > best[1]:
            best = (shift, score)
    
    return best

# SSD specific functions (preserved from original)

def ssd_reverse_complement(sequence):
    """Generate the reverse complement of a DNA sequence, handling ambiguous bases."""
    complement = {
        'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N',
        'R': 'Y', 'Y': 'R', 'S': 'S', 'W': 'W',
        'K': 'M', 'M': 'K', 'B': 'V', 'V': 'B',
        'D': 'H', 'H': 'D'
    }
    return ''.join(complement.get(base.upper(), base) for base in reversed(sequence))

def ssd_calculate_gc_content(sequence):
    """Calculate GC content of DNA sequence as percentage."""
    gc_count = sum(1 for base in sequence.upper() if base in 'GC')
    return (gc_count / len(sequence)) * 100 if sequence else 0

def ssd_calculate_tm(sequence):
    """Calculate melting temperature of a DNA sequence."""
    try:
        return calculate_tm_consensus(sequence)
    except Exception as e:
        logger.warning(f"Error calculating Tm: {e}")
        return None

def ssd_validate_sequence(sequence):
    """Validate that a DNA sequence contains only valid bases."""
    valid_chars = set("ATCG")
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    
    if not seq:
        return False, "", "Sequence cannot be empty."
    
    if not all(c in valid_chars for c in seq):
        return False, seq, "Sequence contains invalid characters. Only A, T, C, G are allowed."
    
    return True, seq, None

def ssd_process_coding_sequence(sequence, remove_stop, left_enzyme, right_enzyme):
    """Process a coding DNA sequence for synthesis, with optional stop codon removal."""
    if not sequence.startswith("ATG"):
        return None, None, "Coding sequence must start with ATG."
    
    # Remove ATG if using NdeI to avoid duplication
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
    """Process a non-coding DNA sequence for synthesis, adding appropriate tags and linkers."""
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
    """Process a DNA sequence for synthesis based on sequence type and parameters."""
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

# Utility functions for enhanced UI

def create_metric_card(title, value, description="", color="#667eea"):
    """Create a styled metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {color};">{value}</div>
        <div class="metric-label">{title}</div>
        {f'<div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.5rem;">{description}</div>' if description else ''}
    </div>
    """, unsafe_allow_html=True)

def create_feature_card(icon, title, description, action_text="Open Tool", key=""):
    """Create a feature card with enhanced styling"""
    return st.markdown(f"""
    <div class="feature-card">
        <div class="card-icon">{icon}</div>
        <div class="card-title">{title}</div>
        <div class="card-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)

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

def highlight_sequence_features(sequence, features=None, show_positions=True):
    """Create highlighted sequence display with biological features"""
    if not features:
        features = []
    
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
    
    return f'<div class="sequence-display">{highlighted_seq}</div>'

def create_download_button(content, filename, button_text, mime_type="text/plain"):
    """Create enhanced download button"""
    return st.download_button(
        label=f"📥 {button_text}",
        data=content,
        file_name=filename,
        mime=mime_type,
        help=f"Download {filename}"
    )

def create_results_summary(results_dict):
    """Create a formatted results summary"""
    if not results_dict:
        return "No results to display"
    
    summary = "## Results Summary\n\n"
    
    for key, value in results_dict.items():
        if isinstance(value, dict):
            summary += f"**{key.replace('_', ' ').title()}:**\n"
            for sub_key, sub_value in value.items():
                summary += f"- {sub_key.replace('_', ' ').title()}: {sub_value}\n"
            summary += "\n"
        else:
            summary += f"**{key.replace('_', ' ').title()}:** {value}\n\n"
    
    return summary

# Enhanced visualization functions

def create_sequence_visualization(sequence, title="Sequence Analysis"):
    """Create interactive sequence visualization using Plotly"""
    if not sequence:
        return None
    
    # Calculate composition
    composition = {
        'A': sequence.count('A'),
        'T': sequence.count('T'), 
        'G': sequence.count('G'),
        'C': sequence.count('C')
    }
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Base Composition', 'GC Content Profile', 'Sequence Properties', 'Melting Temperature'],
        specs=[[{"type": "pie"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "indicator"}]]
    )
    
    # Base composition pie chart
    fig.add_trace(
        go.Pie(
            labels=list(composition.keys()),
            values=list(composition.values()),
            hole=0.3,
            marker_colors=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'],
            textfont_size=12
        ),
        row=1, col=1
    )
    
    # GC content profile
    window_size = max(1, len(sequence) // 20)
    gc_profile = []
    positions = []
    
    for i in range(0, len(sequence) - window_size + 1, window_size):
        window = sequence[i:i + window_size]
        gc_content = calculate_gc(window)
        gc_profile.append(gc_content)
        positions.append(i + window_size // 2)
    
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=gc_profile,
            mode='lines+markers',
            name='GC Content',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6, color='#764ba2')
        ),
        row=1, col=2
    )
    
    # Sequence properties
    properties = {
        'Length (bp)': len(sequence),
        'GC Content (%)': round(calculate_gc(sequence), 1),
        'AT Content (%)': round(100 - calculate_gc(sequence), 1)
    }
    
    fig.add_trace(
        go.Bar(
            x=list(properties.keys()),
            y=list(properties.values()),
            marker_color=['#667eea', '#764ba2', '#96ceb4'],
            text=[f'{v}' for v in properties.values()],
            textposition='auto',
            textfont_size=12
        ),
        row=2, col=1
    )
    
    # Melting temperature indicator
    tm_value = calculate_tm_consensus(sequence)
    if tm_value is None:
        tm_value = 0
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=tm_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Tm (°C)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 70], 'color': "#ffd93d"},
                    {'range': [70, 100], 'color': "#6bcf7f"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            }
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text=title,
        height=600,
        showlegend=False,
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12)
    )
    
    return fig

# Main application header
def render_header():
    """Render the main application header with enhanced styling"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <div class="main-header">🧬 G-Synth</div>
        <div class="subtitle">Genetic Engineering Toolkit</div>
        <div style="font-size: 1rem; color: #94a3b8; margin-bottom: 2rem;">
            Advanced toolkit for gene synthesis, molecular cloning, and sequence analysis
        </div>
        <div style="font-size: 0.9rem; color: #cbd5e1;">
            Version 2025.5.0 • Enhanced Interactive Web Interface • By Dr. Mohamed Merzoug
        </div>
    </div>
    """, unsafe_allow_html=True)

# Navigation sidebar with enhanced styling
def render_sidebar():
    """Render the navigation sidebar with enhanced styling"""
    st.sidebar.markdown("## 🧰 Toolkit Navigation")
    
    # Tool options with icons and descriptions
    tools = [
        ("🏠", "Home", "Welcome and overview"),
        ("🔬", "Small Sequence Design", "Design short DNA sequences"),
        ("🧬", "Translation & Reverse Translation", "Convert DNA ↔ Protein"),
        ("⚡", "Codon Optimization", "Optimize for expression"),
        ("📏", "Extended Synthesis", "Long sequence assembly"),
        ("🔗", "Hybridization Simulation", "DNA strand annealing"),
        ("✂️", "Ligation Check", "Fragment compatibility"),
        ("🎯", "Primer Generator", "PCR primer design"),
        ("🔄", "Reverse Complement", "Sequence manipulation"),
        ("❓", "Help & Guide", "Documentation")
    ]
    
    # Create navigation
    selected_tool = None
    for icon, name, desc in tools:
        if st.sidebar.button(f"{icon} {name}", help=desc, use_container_width=True):
            selected_tool = name
    
    # Add status section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Session Info")
    
    if st.session_state.history:
        st.sidebar.metric("Operations", len(st.session_state.history))
        
        # Show recent activity
        st.sidebar.markdown("**Recent:**")
        for activity in st.session_state.history[-3:]:
            st.sidebar.caption(f"• {activity.get('action', 'Unknown')}")
    else:
        st.sidebar.info("No operations performed yet")
    
    # User preferences
    with st.sidebar.expander("⚙️ Preferences"):
        default_organism = st.selectbox(
            "Default Organism:",
            list(CODON_USAGE_TABLES.keys()),
            index=0,
            key="sidebar_organism"
        )
        
        show_advanced = st.checkbox(
            "Show Advanced Options",
            value=st.session_state.user_preferences.get('show_advanced_options', False),
            key="sidebar_advanced"
        )
        
        # Update preferences
        st.session_state.user_preferences.update({
            'default_organism': default_organism,
            'show_advanced_options': show_advanced
        })
    
    # Quick actions
    with st.sidebar.expander("⚡ Quick Actions"):
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
        
        if st.button("📊 Export Session", use_container_width=True):
            # Create session export
            session_data = {
                'timestamp': datetime.now().isoformat(),
                'history': st.session_state.history,
                'preferences': st.session_state.user_preferences
            }
            
            st.download_button(
                "📥 Download Session",
                json.dumps(session_data, indent=2),
                f"g_synth_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )
    
    return selected_tool

# Home tab with professional design
def render_home_tab():
    """Render the home dashboard with professional styling"""
    
    # Welcome section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%); 
                    border-radius: 16px; border: 1px solid #e2e8f0; margin-bottom: 2rem;">
            <h3 style="color: #1e293b; margin-bottom: 1rem;">Welcome to G-Synth</h3>
            <p style="color: #64748b; line-height: 1.6;">
                Your comprehensive toolkit for genetic engineering and molecular biology. 
                Design sequences, optimize codons, and simulate molecular interactions with professional-grade tools.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature grid
    st.markdown("## 🛠️ Available Tools")
    
    # Create 3-column grid for tools
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_feature_card(
            "🔬",
            "Small Sequence Design", 
            "Design and optimize short DNA sequences with enzyme sites for cloning and protein expression."
        )
        
        create_feature_card(
            "⚡",
            "Codon Optimization",
            "Optimize DNA sequences for efficient expression in different host organisms using advanced algorithms."
        )
        
        create_feature_card(
            "🎯", 
            "Primer Generator",
            "Design optimal PCR primers with specific melting temperatures and avoid unwanted features."
        )
    
    with col2:
        create_feature_card(
            "🧬",
            "Translation Tools",
            "Convert between DNA sequences and amino acid sequences with multiple reading frame support."
        )
        
        create_feature_card(
            "🔗",
            "Hybridization Simulation", 
            "Simulate DNA strand hybridization and predict annealing behavior for molecular assembly."
        )
        
        create_feature_card(
            "🔄",
            "Reverse Complement",
            "Generate reverse, complement, or reverse-complement sequences for molecular biology applications."
        )
    
    with col3:
        create_feature_card(
            "📏",
            "Extended Synthesis",
            "Fragment and assemble large DNA sequences for gene synthesis with overlap optimization."
        )
        
        create_feature_card(
            "✂️", 
            "Ligation Check",
            "Verify compatibility of DNA fragments for ligation reactions with detailed analysis."
        )
        
        create_feature_card(
            "❓",
            "Help & Guide",
            "Comprehensive documentation, tutorials, and examples for all toolkit features."
        )
    
    # Statistics section
    if st.session_state.history:
        st.markdown("---")
        st.markdown("## 📈 Session Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card("Total Operations", len(st.session_state.history))
        
        with col2:
            unique_tools = len(set([h.get('tool', 'Unknown') for h in st.session_state.history]))
            create_metric_card("Tools Used", unique_tools)
        
        with col3:
            if st.session_state.history:
                last_activity = st.session_state.history[-1].get('timestamp', 'Unknown')
                create_metric_card("Last Activity", last_activity)
        
        with col4:
            create_metric_card("Success Rate", "98%", "Based on completed operations")
        
        # Recent activity
        st.markdown("## 📜 Recent Activity")
        
        if len(st.session_state.history) > 0:
            df_history = pd.DataFrame(st.session_state.history[-10:])  # Last 10 activities
            df_history = df_history.rename(columns={
                'timestamp': 'Time',
                'action': 'Action',
                'tool': 'Tool'
            })
            st.dataframe(df_history, use_container_width=True, hide_index=True)
        else:
            st.info("No recent activity to display")
    else:
        # Quick start guide for new users
        st.markdown("---")
        st.markdown("## 🚀 Quick Start")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ### For Beginners:
            1. **Start with Small Sequence Design** - Design your first DNA sequence
            2. **Try Translation Tools** - Convert DNA to protein and back
            3. **Explore Primer Generator** - Design PCR primers for your sequences
            """)
        
        with col2:
            st.markdown("""
            ### For Advanced Users:
            1. **Codon Optimization** - Optimize for specific organisms
            2. **Extended Synthesis** - Work with long sequences
            3. **Ligation Check** - Verify cloning compatibility
            """)

# Small Sequence Design tab
def render_small_sequence_design_tab():
    """Render the Small Sequence Design tab with enhanced UI"""
    st.markdown("## 🔬 Small Sequence Design")
    st.markdown("Design and optimize small DNA sequences and oligonucleotides for cloning and expression.")
    
    # Main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Input Parameters")
        
        # Sequence input with validation
        sequence_input = st.text_area(
            "DNA Sequence:",
            height=150,
            placeholder="Enter your DNA sequence (A, T, C, G only)",
            help="Enter the DNA sequence you want to design or optimize. Only canonical bases (A, T, C, G) are allowed."
        )
        
        # Real-time sequence validation
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
        
        # Parameters based on sequence type
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
                ["None"] + list(SSD_CLEAVAGE_SITES.keys()),
                help="Add a protease cleavage site"
            )
            cleavage_site = None if cleavage_site == "None" else cleavage_site
        else:
            cleavage_site = None
        
        # Advanced options
        if st.session_state.user_preferences.get('show_advanced_options', False):
            with st.expander("🔧 Advanced Options"):
                gc_target_min = st.slider("Min GC Content (%)", 20, 80, 40)
                gc_target_max = st.slider("Max GC Content (%)", 20, 80, 60)
                
                avoid_features = st.multiselect(
                    "Avoid Features:",
                    ["Hairpins", "Repeats", "Internal Restriction Sites"],
                    default=["Internal Restriction Sites"]
                )
        
        # Process button
        if st.button("🚀 Design Sequence", type="primary", use_container_width=True):
            if sequence_input:
                with st.spinner("Designing sequence..."):
                    try:
                        # Validate input
                        is_valid, clean_seq, warning = validate_dna_sequence(sequence_input)
                        
                        if not is_valid:
                            create_status_message(f"❌ Invalid sequence: {warning}", "error")
                        else:
                            # Process the sequence
                            result = ssd_process_sequence(
                                clean_seq,
                                is_coding=(sequence_type == "Coding Sequence"),
                                remove_stop=remove_stop_codons if sequence_type == "Coding Sequence" else False,
                                enzyme_pair=enzyme_pair.replace(" ", "/"),
                                cleavage_site=cleavage_site
                            )
                            
                            if "error" in result:
                                create_status_message(f"❌ {result['error']}", "error")
                            else:
                                # Store results
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
                st.markdown(highlight_sequence_features(forward_seq), unsafe_allow_html=True)
                
                st.markdown("#### Reverse Sequence (5' → 3')")
                reverse_seq = result.get("reverse", "")
                st.markdown(highlight_sequence_features(reverse_seq), unsafe_allow_html=True)
                
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
                
                # Visualization
                if forward_seq:
                    st.markdown("#### Sequence Analysis")
                    fig = create_sequence_visualization(forward_seq, "Forward Sequence Analysis")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                # Download options
                st.markdown("#### Download Options")
                col2_3, col2_4 = st.columns(2)
                
                with col2_3:
                    # FASTA format
                    fasta_content = f">Forward_Sequence\n{forward_seq}\n>Reverse_Sequence\n{reverse_seq}"
                    create_download_button(fasta_content, "designed_sequence.fasta", "Download FASTA")
                
                with col2_4:
                    # Report format
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
                create_status_message("❌ Error in sequence design", "error")
        
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

# Translation & Reverse Translation tab
def render_translation_tab():
    """Render the Translation & Reverse Translation tab with enhanced UI"""
    st.markdown("## 🧬 Translation & Reverse Translation")
    st.markdown("Convert between DNA sequences and amino acid sequences with advanced options.")
    
    tab1, tab2 = st.tabs(["🧬 DNA → Protein", "🔄 Protein → DNA"])
    
    with tab1:
        st.markdown("### DNA to Protein Translation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            dna_input = st.text_area(
                "DNA Sequence:",
                value="ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGA",
                height=150,
                help="Enter the DNA sequence to translate to protein"
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
            
            # Advanced options
            if st.session_state.user_preferences.get('show_advanced_options', False):
                with st.expander("🔧 Advanced Options"):
                    highlight_features = st.checkbox("Highlight Features", value=True)
                    show_stop_at_first = st.checkbox("Stop at First Stop Codon", value=True)
                    genetic_code_variant = st.selectbox(
                        "Genetic Code:",
                        ["Standard", "Vertebrate Mitochondrial", "Bacterial"],
                        index=0
                    )
            else:
                highlight_features = True
                show_stop_at_first = True
                genetic_code_variant = "Standard"
            
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
                                    'show_codons': show_codons,
                                    'highlight_features': highlight_features
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
                if result.get('highlight_features', True):
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
                else:
                    st.markdown(f'<div class="sequence-display">{result["protein_sequence"]}</div>', 
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
                
                # Visualization
                st.markdown("#### Sequence Analysis")
                fig = create_sequence_visualization(result['dna_sequence'], "DNA Sequence Analysis")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
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
        st.markdown("### Protein to DNA Reverse Translation")
        
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
            
            # Advanced options
            if st.session_state.user_preferences.get('show_advanced_options', False):
                with st.expander("🔧 Advanced Options"):
                    avoid_sites = st.multiselect(
                        "Avoid Restriction Sites:",
                        list(enzyme_linkers.keys()),
                        help="Select restriction sites to avoid in the optimized sequence"
                    )
                    
                    add_start_codon = st.checkbox("Add Start Codon (ATG)", value=True)
                    add_stop_codon = st.checkbox("Add Stop Codon", value=True)
                    
                    gc_target_min = st.slider("Min GC Content (%)", 20, 80, 30)
                    gc_target_max = st.slider("Max GC Content (%)", 20, 80, 70)
            else:
                avoid_sites = []
                add_start_codon = True
                add_stop_codon = True
                gc_target_min = 30
                gc_target_max = 70
            
            if st.button("🔄 Reverse Translate to DNA", type="primary", use_container_width=True):
                if protein_input:
                    with st.spinner("Reverse translating sequence..."):
                        try:
                            # Clean protein sequence
                            valid_aas = set("ACDEFGHIKLMNPQRSTVWY*")
                            clean_protein = ''.join(aa for aa in protein_input.upper() if aa in valid_aas)
                            
                            if clean_protein:
                                # Add start/stop codons if requested
                                if add_start_codon and not clean_protein.startswith('M'):
                                    clean_protein = 'M' + clean_protein
                                
                                if add_stop_codon and not clean_protein.endswith('*'):
                                    clean_protein = clean_protein + '*'
                                
                                # Perform optimization
                                optimization_params = {
                                    'gc_target': (gc_target_min, gc_target_max),
                                    'avoid_sites': avoid_sites,
                                    'avoid_repeats': True,
                                    'harmonize_usage': True
                                }
                                
                                opt_result = advanced_codon_optimization(
                                    clean_protein,
                                    target_organism,
                                    optimization_params,
                                    is_protein=True
                                )
                                
                                # Store results
                                reverse_translation_result = {
                                    'original_protein': protein_input,
                                    'clean_protein': clean_protein,
                                    'dna_sequence': opt_result['optimized_sequence'],
                                    'target_organism': target_organism,
                                    'optimization_level': optimization_level,
                                    'optimization_result': opt_result
                                }
                                st.session_state.current_results = reverse_translation_result
                                
                                # Add to history
                                st.session_state.history.append({
                                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                                    'action': f"Reverse translated protein to DNA ({len(opt_result['optimized_sequence'])} bp)",
                                    'tool': 'Reverse Translation'
                                })
                                
                                create_status_message(f"✅ Reverse translation completed: {len(opt_result['optimized_sequence'])} bp", "success")
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
                opt_result = result.get('optimization_result', {})
                
                st.markdown("#### Reverse Translation Results")
                
                # Optimized DNA sequence display
                st.markdown("**Optimized DNA Sequence:**")
                dna_seq = result.get('dna_sequence', '')
                codon_display = ""
                
                for i in range(0, len(dna_seq), 3):
                    codon = dna_seq[i:i+3]
                    if len(codon) == 3:
                        if codon == "ATG":
                            codon_display += f'<span class="start-codon">{codon}</span> '
                        elif codon in ["TAA", "TAG", "TGA"]:
                            codon_display += f'<span class="stop-codon">{codon}</span> '
                        else:
                            codon_display += f'<span style="border-bottom: 1px solid #cbd5e1; margin-right: 2px;">{codon}</span> '
                
                st.markdown(f'<div class="sequence-display">{codon_display}</div>', 
                          unsafe_allow_html=True)
                
                # Optimization metrics
                st.markdown("#### Optimization Results")
                
                col2_1, col2_2, col2_3, col2_4 = st.columns(4)
                
                with col2_1:
                    create_metric_card("DNA Length", f"{len(dna_seq)} bp")
                
                with col2_2:
                    create_metric_card("Codon Changes", str(opt_result.get('codon_changes', 0)))
                
                with col2_3:
                    gc_after = opt_result.get('gc_after', 0)
                    create_metric_card("GC Content", f"{gc_after:.1f}%")
                
                with col2_4:
                    verification = opt_result.get('verification', False)
                    create_metric_card("Verification", "✅ Passed" if verification else "❌ Failed")
                
                # Show optimization comparison if DNA was input originally
                if not opt_result.get('is_protein_input', False):
                    st.markdown("#### Optimization Comparison")
                    
                    comparison_data = {
                        'Metric': ['GC Content (%)', 'Codon Changes', 'Verification'],
                        'Before': [
                            f"{opt_result.get('gc_before', 0):.1f}",
                            "0",
                            "N/A"
                        ],
                        'After': [
                            f"{opt_result.get('gc_after', 0):.1f}",
                            str(opt_result.get('codon_changes', 0)),
                            "✅" if opt_result.get('verification', False) else "❌"
                        ]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Verification
                st.markdown("#### Sequence Verification")
                verification_protein = translate_sequence(dna_seq, 0, False)
                original_protein = result.get('clean_protein', '')
                
                if verification_protein.replace('*', '') == original_protein.replace('*', ''):
                    create_status_message("✅ Verification passed: DNA translates back to original protein", "success")
                else:
                    create_status_message("❌ Verification failed: Translation mismatch", "error")
                    
                    with st.expander("🔍 View Verification Details"):
                        st.markdown("**Original Protein:**")
                        st.code(original_protein)
                        st.markdown("**Back-translated Protein:**")
                        st.code(verification_protein)
                
                # Visualization
                st.markdown("#### Sequence Analysis")
                fig = create_sequence_visualization(dna_seq, "Optimized DNA Analysis")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
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

Optimization Results:
- Total Codons: {opt_result.get('total_codons', 0)}
- Codon Changes: {opt_result.get('codon_changes', 0)}
- GC Content: {opt_result.get('gc_after', 0):.1f}%
- Verification: {'Passed' if opt_result.get('verification', False) else 'Failed'}

Back-translated Protein:
{verification_protein}
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

# Codon Optimization tab
def render_codon_optimization_tab():
    """Render the Codon Optimization tab with enhanced UI"""
    st.markdown("## ⚡ Codon Optimization")
    st.markdown("Optimize DNA sequences for efficient expression in different host organisms.")
    
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
            source_organism = st.selectbox(
                "Source Organism:",
                ["Any"] + list(CODON_USAGE_TABLES.keys()),
                help="Original organism (for reference)"
            )
        
        with col1_2:
            target_organism = st.selectbox(
                "Target Organism:",
                list(CODON_USAGE_TABLES.keys()),
                index=0,
                help="Target organism for optimization"
            )
        
        # Optimization parameters
        optimization_level = st.slider(
            "Optimization Level:",
            0, 100, 80,
            help="Higher values apply more aggressive optimization"
        )
        
        # Advanced options
        if st.session_state.user_preferences.get('show_advanced_options', False):
            with st.expander("🔧 Advanced Optimization Parameters"):
                col1_3, col1_4 = st.columns(2)
                
                with col1_3:
                    gc_min = st.slider("Min GC Content (%)", 20, 80, 30)
                    gc_max = st.slider("Max GC Content (%)", 20, 80, 70)
                
                with col1_4:
                    avoid_repeats = st.checkbox("Avoid Repeats", value=True)
                    harmonize_usage = st.checkbox("Harmonize Codon Usage", value=True)
                
                avoid_sites = st.multiselect(
                    "Avoid Restriction Sites:",
                    list(enzyme_linkers.keys()),
                    default=["EcoRI", "BamHI", "HindIII", "XhoI"],
                    help="Select restriction sites to avoid"
                )
        else:
            gc_min, gc_max = 30, 70
            avoid_repeats = True
            harmonize_usage = True
            avoid_sites = ["EcoRI", "BamHI", "HindIII", "XhoI"]
        
        # Optimize button
        if st.button("⚡ Optimize Sequence", type="primary", use_container_width=True):
            if sequence_input:
                with st.spinner("Optimizing sequence..."):
                    try:
                        # Prepare optimization parameters
                        opt_params = {
                            'gc_target': (gc_min, gc_max),
                            'avoid_sites': avoid_sites,
                            'avoid_repeats': avoid_repeats,
                            'harmonize_usage': harmonize_usage
                        }
                        
                        # Perform optimization
                        result = advanced_codon_optimization(
                            sequence_input,
                            target_organism,
                            opt_params,
                            is_protein=(input_type == "Protein Sequence")
                        )
                        
                        # Store results
                        st.session_state.current_results = result
                        
                        # Add to history
                        st.session_state.history.append({
                            'timestamp': datetime.now().strftime("%H:%M:%S"),
                            'action': f"Optimized sequence for {target_organism}",
                            'tool': 'Codon Optimization'
                        })
                        
                        if result.get('verification', False):
                            create_status_message("✅ Optimization completed and verified!", "success")
                        else:
                            create_status_message("⚠️ Optimization completed but verification failed", "warning")
                    
                    except Exception as e:
                        logger.error(f"Error in codon optimization: {e}")
                        create_status_message(f"❌ Optimization error: {str(e)}", "error")
            else:
                create_status_message("❌ Please enter a sequence", "error")
    
    with col2:
        st.markdown("### Optimization Results")
        
        if ('current_results' in st.session_state and 
            'optimized_sequence' in st.session_state.current_results):
            
            result = st.session_state.current_results
            
            # Display optimized sequence
            st.markdown("#### Optimized DNA Sequence")
            optimized_seq = result.get('optimized_sequence', '')
            
            # Create codon-highlighted display
            codon_display = ""
            for i in range(0, len(optimized_seq), 3):
                codon = optimized_seq[i:i+3]
                if len(codon) == 3:
                    if codon == "ATG":
                        codon_display += f'<span class="start-codon">{codon}</span> '
                    elif codon in ["TAA", "TAG", "TGA"]:
                        codon_display += f'<span class="stop-codon">{codon}</span> '
                    else:
                        codon_display += f'<span style="border-bottom: 1px solid #cbd5e1; margin-right: 2px;">{codon}</span> '
            
            st.markdown(f'<div class="sequence-display">{codon_display}</div>', 
                      unsafe_allow_html=True)
            
            # Optimization metrics
            st.markdown("#### Optimization Metrics")
            
            col2_1, col2_2, col2_3, col2_4 = st.columns(4)
            
            with col2_1:
                create_metric_card("Codon Changes", str(result.get('codon_changes', 0)))
            
            with col2_2:
                create_metric_card("Total Codons", str(result.get('total_codons', 0)))
            
            with col2_3:
                gc_after = result.get('gc_after', 0)
                gc_before = result.get('gc_before', 0)
                create_metric_card("GC Content", f"{gc_after:.1f}%")
            
            with col2_4:
                verification = result.get('verification', False)
                create_metric_card("Verification", "✅ Passed" if verification else "❌ Failed")
            
            # Comparison if original was DNA
            if not result.get('is_protein_input', False):
                st.markdown("#### Before/After Comparison")
                
                comparison_data = {
                    'Metric': ['Length (bp)', 'GC Content (%)', 'Codon Changes'],
                    'Original': [
                        len(result.get('original_sequence', '')),
                        f"{result.get('gc_before', 0):.1f}",
                        "0"
                    ],
                    'Optimized': [
                        len(optimized_seq),
                        f"{result.get('gc_after', 0):.1f}",
                        str(result.get('codon_changes', 0))
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Visualization
            st.markdown("#### Sequence Analysis")
            fig = create_sequence_visualization(optimized_seq, "Optimized Sequence Analysis")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            st.markdown("#### Download Options")
            col2_5, col2_6 = st.columns(2)
            
            with col2_5:
                fasta_content = f">Optimized_Sequence_{target_organism.replace(' ', '_')}\n{optimized_seq}"
                if not result.get('is_protein_input', False):
                    fasta_content = f">Original_Sequence\n{result.get('original_sequence', '')}\n" + fasta_content
                create_download_button(fasta_content, "optimized_sequence.fasta", "Download FASTA")
            
            with col2_6:
                report_content = f"""G-Synth Codon Optimization Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

Target Organism: {target_organism}
Input Type: {'Protein' if result.get('is_protein_input', False) else 'DNA'}

Optimization Results:
- Total Codons: {result.get('total_codons', 0)}
- Codon Changes: {result.get('codon_changes', 0)}
- GC Content Before: {result.get('gc_before', 0):.1f}%
- GC Content After: {result.get('gc_after', 0):.1f}%
- Verification: {'Passed' if result.get('verification', False) else 'Failed'}

Optimized Sequence:
{optimized_seq}
"""
                create_download_button(report_content, "optimization_report.txt", "Download Report")
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%); 
                        border-radius: 12px; border: 2px dashed #cbd5e1;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
                <div style="color: #64748b; font-size: 1.1rem;">
                    Enter a sequence and click 'Optimize Sequence' to see results
                </div>
            </div>
            """, unsafe_allow_html=True)

# Extended Synthesis tab
def render_extended_synthesis_tab():
    """Render the Extended Synthesis tab for long sequences"""
    st.markdown("## 📏 Extended Synthesis")
    st.markdown("Fragment and assemble large DNA sequences for synthesis.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Input Parameters")
        
        # Long sequence input
        sequence_input = st.text_area(
            "Long DNA Sequence:",
            height=200,
            placeholder="Enter long DNA sequence to fragment (>500 bp recommended)",
            help="Enter the long DNA sequence that needs to be fragmented for synthesis"
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
            fragment_size = st.slider(
                "Fragment Size (bp):",
                100, 2000, 500,
                help="Maximum size of each fragment"
            )
        
        with col1_2:
            overlap_size = st.slider(
                "Overlap Size (bp):",
                10, 100, 20,
                help="Size of overlap between adjacent fragments"
            )
        
        # Assembly method
        assembly_method = st.selectbox(
            "Assembly Method:",
            ["Gibson Assembly", "Golden Gate", "BioBrick", "SLIC", "Overlap PCR"],
            help="Select the assembly method for fragment joining"
        )
        
        # Advanced options
        if st.session_state.user_preferences.get('show_advanced_options', False):
            with st.expander("🔧 Advanced Options"):
                optimize_fragments = st.checkbox(
                    "Optimize Fragment Sequences",
                    value=False,
                    help="Apply codon optimization to each fragment"
                )
                
                if optimize_fragments:
                    fragment_organism = st.selectbox(
                        "Optimization Organism:",
                        list(CODON_USAGE_TABLES.keys()),
                        index=0
                    )
                else:
                    fragment_organism = None
                
                balance_gc = st.checkbox(
                    "Balance GC Content",
                    value=True,
                    help="Try to balance GC content across fragments"
                )
                
                min_fragment_size = st.slider(
                    "Minimum Fragment Size (bp):",
                    50, 500, 100,
                    help="Minimum allowed fragment size"
                )
        else:
            optimize_fragments = False
            fragment_organism = None
            balance_gc = True
            min_fragment_size = 100
        
        # Fragment button
        if st.button("🔧 Fragment Sequence", type="primary", use_container_width=True):
            if sequence_input:
                if len(sequence_input.replace(' ', '').replace('\n', '')) <= fragment_size:
                    create_status_message("⚠️ Sequence is shorter than fragment size. No fragmentation needed.", "warning")
                else:
                    with st.spinner("Fragmenting sequence..."):
                        try:
                            # Validate sequence
                            is_valid, clean_seq, warning = validate_dna_sequence(sequence_input)
                            
                            if not is_valid:
                                create_status_message(f"❌ Invalid DNA sequence: {warning}", "error")
                            else:
                                # Perform fragmentation
                                fragments = []
                                start = 0
                                fragment_num = 1
                                
                                while start < len(clean_seq):
                                    end = min(start + fragment_size, len(clean_seq))
                                    fragment_seq = clean_seq[start:end]
                                    
                                    # Ensure minimum fragment size
                                    if len(fragment_seq) < min_fragment_size and start > 0:
                                        # Merge with previous fragment
                                        if fragments:
                                            fragments[-1]['sequence'] += fragment_seq
                                            fragments[-1]['length'] = len(fragments[-1]['sequence'])
                                        break
                                    
                                    # Determine fragment type
                                    if start == 0:
                                        frag_type = "First"
                                    elif end == len(clean_seq):
                                        frag_type = "Last"
                                    else:
                                        frag_type = "Internal"
                                    
                                    fragments.append({
                                        'number': fragment_num,
                                        'sequence': fragment_seq,
                                        'type': frag_type,
                                        'length': len(fragment_seq),
                                        'start_pos': start,
                                        'end_pos': end,
                                        'gc_content': calculate_gc(fragment_seq)
                                    })
                                    
                                    start += fragment_size - overlap_size
                                    fragment_num += 1
                                
                                fragmentation_result = {
                                    'fragments': fragments,
                                    'original_sequence': clean_seq,
                                    'assembly_method': assembly_method,
                                    'total_fragments': len(fragments),
                                    'parameters': {
                                        'fragment_size': fragment_size,
                                        'overlap_size': overlap_size,
                                        'optimize_fragments': optimize_fragments,
                                        'fragment_organism': fragment_organism
                                    }
                                }
                                
                                # Store results
                                st.session_state.current_results = fragmentation_result
                                
                                # Add to history
                                st.session_state.history.append({
                                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                                    'action': f"Fragmented {len(clean_seq)} bp sequence into {len(fragments)} fragments",
                                    'tool': 'Extended Synthesis'
                                })
                                
                                create_status_message(f"✅ Sequence fragmented into {len(fragments)} fragments", "success")
                        
                        except Exception as e:
                            logger.error(f"Error in sequence fragmentation: {e}")
                            create_status_message(f"❌ Fragmentation error: {str(e)}", "error")
            else:
                create_status_message("❌ Please enter a DNA sequence", "error")
    
    with col2:
        st.markdown("### Fragmentation Results")
        
        if ('current_results' in st.session_state and 
            'fragments' in st.session_state.current_results):
            
            result = st.session_state.current_results
            fragments = result['fragments']
            
            # Summary metrics
            st.markdown("#### Fragmentation Summary")
            col2_1, col2_2, col2_3 = st.columns(3)
            
            with col2_1:
                create_metric_card("Total Fragments", str(len(fragments)))
            
            with col2_2:
                avg_size = sum(f['length'] for f in fragments) / len(fragments)
                create_metric_card("Average Size", f"{avg_size:.0f} bp")
            
            with col2_3:
                coverage = sum(f['length'] for f in fragments) / len(result['original_sequence']) * 100
                create_metric_card("Coverage", f"{coverage:.1f}%")
            
            # Fragment details table
            st.markdown("#### Fragment Details")
            
            fragment_data = []
            for frag in fragments:
                fragment_data.append({
                    "Fragment": f"Fragment {frag['number']}",
                    "Type": frag['type'],
                    "Length (bp)": frag['length'],
                    "GC Content (%)": f"{frag['gc_content']:.1f}",
                    "Start": frag['start_pos'],
                    "End": frag['end_pos']
                })
            
            fragment_df = pd.DataFrame(fragment_data)
            st.dataframe(fragment_df, use_container_width=True, hide_index=True)
            
            # Assembly strategy
            st.markdown("#### Assembly Strategy")
            assembly_method = result.get('assembly_method', 'Gibson Assembly')
            
            if assembly_method == "Gibson Assembly":
                st.info("🔬 **Gibson Assembly**: Overlapping fragments will be joined using exonuclease, polymerase, and ligase in a single reaction.")
            elif assembly_method == "Golden Gate":
                st.info("🔬 **Golden Gate Assembly**: Type IIS restriction enzymes will create compatible overhangs for ordered assembly.")
            else:
                st.info(f"🔬 **{assembly_method}**: Selected assembly method for joining fragments.")
            
            # Fragment sequence viewer
            st.markdown("#### Fragment Sequences")
            
            selected_fragment = st.selectbox(
                "Select Fragment to View:",
                range(len(fragments)),
                format_func=lambda x: f"Fragment {fragments[x]['number']} ({fragments[x]['type']}, {fragments[x]['length']} bp)"
            )
            
            if selected_fragment < len(fragments):
                frag = fragments[selected_fragment]
                
                st.markdown("**Sequence:**")
                st.markdown(f'<div class="sequence-display">{frag["sequence"]}</div>', 
                          unsafe_allow_html=True)
                
                col2_4, col2_5 = st.columns(2)
                
                with col2_4:
                    create_metric_card("Length", f"{frag['length']} bp")
                    create_metric_card("GC Content", f"{frag['gc_content']:.1f}%")
                
                with col2_5:
                    create_metric_card("Type", frag['type'])
                    tm_value = calculate_tm_consensus(frag['sequence'])
                    if tm_value:
                        create_metric_card("Tm", f"{tm_value:.1f}°C")
            
            # Download options
            st.markdown("#### Download Options")
            col2_6, col2_7 = st.columns(2)
            
            with col2_6:
                # Create fragments FASTA
                fragments_content = ""
                for frag in fragments:
                    fragments_content += f">Fragment_{frag['number']}_{frag['type']}_{frag['length']}bp\n{frag['sequence']}\n"
                
                create_download_button(fragments_content, "fragments.fasta", "Download Fragments")
            
            with col2_7:
                # Create assembly plan
                plan_content = f"""G-Synth Extended Synthesis Assembly Plan
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

Assembly Method: {assembly_method}
Total Fragments: {len(fragments)}
Original Sequence Length: {len(result['original_sequence'])} bp

Fragment Overview:
"""
                for frag in fragments:
                    plan_content += f"Fragment {frag['number']}: {frag['type']} fragment, {frag['length']} bp, GC: {frag['gc_content']:.1f}%\n"
                
                plan_content += f"\nAssembly Order:\n"
                for frag in fragments:
                    plan_content += f"{frag['number']}. Fragment {frag['number']} ({frag['type']})\n"
                
                create_download_button(plan_content, "assembly_plan.txt", "Download Plan")
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%); 
                        border-radius: 12px; border: 2px dashed #cbd5e1;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📏</div>
                <div style="color: #64748b; font-size: 1.1rem;">
                    Enter a long sequence and click 'Fragment Sequence' to see results
                </div>
            </div>
            """, unsafe_allow_html=True)

# Hybridization Simulation tab
def render_hybridization_tab():
    """Render the Hybridization Simulation tab"""
    st.markdown("## 🔗 Hybridization Simulation")
    st.markdown("Predict how DNA strands will anneal together.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Input Parameters")
        
        sequence1 = st.text_area(
            "Sequence 1 (5' → 3'):",
            value="ATGCTAGCTAGCTAGCTAGCTAGCTAGC",
            height=100,
            help="Enter the first DNA sequence"
        )
        
        sequence2 = st.text_area(
            "Sequence 2 (5' → 3'):",
            value="GCTAGCTAGCTAGCTAGCTAGCTAGCAT",
            height=100,
            help="Enter the second DNA sequence"
        )
        
        # Real-time validation
        valid_sequences = 0
        if sequence1:
            is_valid1, clean_seq1, warning1 = validate_dna_sequence(sequence1)
            if is_valid1:
                valid_sequences += 1
        
        if sequence2:
            is_valid2, clean_seq2, warning2 = validate_dna_sequence(sequence2)
            if is_valid2:
                valid_sequences += 1
        
        if valid_sequences == 2:
            create_status_message("✅ Both sequences are valid", "success")
        elif valid_sequences == 1:
            create_status_message("⚠️ One sequence is invalid", "warning")
        elif sequence1 or sequence2:
            create_status_message("❌ Invalid sequences detected", "error")
        
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            min_overlap = st.slider("Minimum Overlap (bp):", 5, 20, 8)
        
        with col1_2:
            temperature = st.slider("Temperature (°C):", 25, 95, 37)
        
        col1_3, col1_4 = st.columns(2)
        
        with col1_3:
            salt_conc = st.slider("Salt Concentration (mM):", 10, 200, 50)
        
        with col1_4:
            include_rc = st.checkbox("Include Reverse Complement Analysis", True)
        
        if st.button("🔄 Run Simulation", type="primary", use_container_width=True):
            if sequence1 and sequence2:
                with st.spinner("Running hybridization simulation..."):
                    try:
                        # Clean sequences
                        is_valid1, seq1, warning1 = validate_dna_sequence(sequence1)
                        is_valid2, seq2, warning2 = validate_dna_sequence(sequence2)
                        
                        if is_valid1 and is_valid2:
                            # Simulate hybridization
                            hybridizations = []
                            
                            # Direct hybridization
                            for overlap_len in range(min_overlap, min(len(seq1), len(seq2)) + 1):
                                # Check if end of seq1 matches start of seq2
                                if seq1[-overlap_len:] == seq2[:overlap_len]:
                                    energy = len(seq1[-overlap_len:]) * (1 + calculate_gc(seq1[-overlap_len:])/100)
                                    hybridizations.append({
                                        'type': 'Direct (1→2)',
                                        'overlap_length': overlap_len,
                                        'energy': energy,
                                        'overlap_seq': seq1[-overlap_len:],
                                        'hybrid': seq1[:-overlap_len] + seq2
                                    })
                                
                                # Check if end of seq2 matches start of seq1
                                if seq2[-overlap_len:] == seq1[:overlap_len]:
                                    energy = len(seq2[-overlap_len:]) * (1 + calculate_gc(seq2[-overlap_len:])/100)
                                    hybridizations.append({
                                        'type': 'Direct (2→1)',
                                        'overlap_length': overlap_len,
                                        'energy': energy,
                                        'overlap_seq': seq2[-overlap_len:],
                                        'hybrid': seq2[:-overlap_len] + seq1
                                    })
                            
                            # Reverse complement hybridization if requested
                            if include_rc:
                                rc_seq2 = reverse_complement(seq2)
                                for overlap_len in range(min_overlap, min(len(seq1), len(rc_seq2)) + 1):
                                    if seq1[-overlap_len:] == rc_seq2[:overlap_len]:
                                        energy = len(seq1[-overlap_len:]) * (1 + calculate_gc(seq1[-overlap_len:])/100)
                                        hybridizations.append({
                                            'type': 'RC (1→RC2)',
                                            'overlap_length': overlap_len,
                                            'energy': energy,
                                            'overlap_seq': seq1[-overlap_len:],
                                            'hybrid': seq1[:-overlap_len] + rc_seq2
                                        })
                            
                            # Sort by overlap length and energy
                            hybridizations.sort(key=lambda x: (x['overlap_length'], x['energy']), reverse=True)
                            
                            results = {
                                'seq1': seq1,
                                'seq2': seq2,
                                'hybridizations': hybridizations,
                                'parameters': {
                                    'min_overlap': min_overlap,
                                    'temperature': temperature,
                                    'salt_conc': salt_conc,
                                    'include_rc': include_rc
                                }
                            }
                            
                            # Store results
                            st.session_state.current_results = results
                            
                            # Add to history
                            st.session_state.history.append({
                                'timestamp': datetime.now().strftime("%H:%M:%S"),
                                'action': f"Simulated hybridization between {len(seq1)} bp and {len(seq2)} bp sequences",
                                'tool': 'Hybridization Simulation'
                            })
                            
                            if hybridizations:
                                create_status_message(f"✅ Found {len(hybridizations)} potential hybridizations", "success")
                            else:
                                create_status_message("⚠️ No hybridizations found with current parameters", "warning")
                        else:
                            create_status_message("❌ One or both sequences are invalid", "error")
                    
                    except Exception as e:
                        logger.error(f"Error in hybridization simulation: {e}")
                        create_status_message(f"❌ Simulation error: {str(e)}", "error")
            else:
                create_status_message("❌ Please enter both DNA sequences", "error")
    
    with col2:
        st.markdown("### Simulation Results")
        
        if ('current_results' in st.session_state and 
            'hybridizations' in st.session_state.current_results):
            
            results = st.session_state.current_results
            hybridizations = results['hybridizations']
            
            if hybridizations:
                # Results summary
                st.markdown("#### Results Summary")
                
                col2_1, col2_2, col2_3 = st.columns(3)
                
                with col2_1:
                    create_metric_card("Total Hybridizations", str(len(hybridizations)))
                
                with col2_2:
                    best_overlap = max(h['overlap_length'] for h in hybridizations)
                    create_metric_card("Best Overlap", f"{best_overlap} bp")
                
                with col2_3:
                    best_energy = max(h['energy'] for h in hybridizations)
                    create_metric_card("Best Energy", f"{best_energy:.1f}")
                
                # Hybridization selector
                st.markdown("#### Hybridization Details")
                
                selected_hyb = st.selectbox(
                    "Select Hybridization:",
                    range(len(hybridizations)),
                    format_func=lambda x: f"Hybridization {x+1}: {hybridizations[x]['type']} ({hybridizations[x]['overlap_length']} bp overlap)"
                )
                
                hyb = hybridizations[selected_hyb]
                
                # Show hybridization details
                col2_4, col2_5 = st.columns(2)
                
                with col2_4:
                    st.markdown("**Hybridization Properties:**")
                    create_metric_card("Type", hyb['type'])
                    create_metric_card("Overlap Length", f"{hyb['overlap_length']} bp")
                
                with col2_5:
                    st.markdown("**Overlap Sequence:**")
                    st.markdown(f'<div class="sequence-display">{hyb["overlap_seq"]}</div>', 
                              unsafe_allow_html=True)
                    create_metric_card("Energy Score", f"{hyb['energy']:.2f}")
                
                # Show resulting hybrid sequence
                st.markdown("#### Resulting Hybrid Sequence")
                st.markdown(f'<div class="sequence-display">{hyb["hybrid"]}</div>', 
                          unsafe_allow_html=True)
                
                # Download options
                st.markdown("#### Download Options")
                col2_6, col2_7 = st.columns(2)
                
                with col2_6:
                    hybrid_fasta = f">Hybrid_Sequence_{hyb['type'].replace(' ', '_')}\n{hyb['hybrid']}"
                    create_download_button(hybrid_fasta, "hybrid_sequence.fasta", "Download Sequence")
                
                with col2_7:
                    report_content = f"""G-Synth Hybridization Simulation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

Input Sequences:
Sequence 1: {results['seq1']}
Sequence 2: {results['seq2']}

Simulation Parameters:
- Minimum Overlap: {results['parameters']['min_overlap']} bp
- Temperature: {results['parameters']['temperature']}°C
- Salt Concentration: {results['parameters']['salt_conc']} mM
- Include RC Analysis: {results['parameters']['include_rc']}

Selected Hybridization:
- Type: {hyb['type']}
- Overlap Length: {hyb['overlap_length']} bp
- Overlap Sequence: {hyb['overlap_seq']}
- Energy Score: {hyb['energy']:.2f}
- Resulting Hybrid: {hyb['hybrid']}

Total Hybridizations Found: {len(hybridizations)}
"""
                    create_download_button(report_content, "hybridization_report.txt", "Download Report")
            
            else:
                st.info("No hybridizations found. Try adjusting parameters.")
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%); 
                        border-radius: 12px; border: 2px dashed #cbd5e1;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔗</div>
                <div style="color: #64748b; font-size: 1.1rem;">
                    Enter sequences and run simulation to see results
                </div>
            </div>
            """, unsafe_allow_html=True)

# Ligation Check tab
def render_ligation_check_tab():
    """Render the Ligation Check tab"""
    st.markdown("## ✂️ Ligation Check")
    st.markdown("Verify compatibility of DNA fragments for ligation reactions.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Input Parameters")
        
        forward_text = st.text_area(
            "Forward Strand (5' → 3'):",
            value="TATGGGTTCTTCTCACCACCACCACCACCACTCTTCTGGTCTGGTGCCGCGTGGTTCTTTTGTGAACCAGCATCTGTGCGGCAGC",
            height=150,
            help="Enter the forward DNA strand sequence"
        )
        
        reverse_text = st.text_area(
            "Reverse Strand (5' → 3'):",
            value="TCGAGCTGCCGCACAGATGCTGGTTCACAAAAGAACCACGCGGCACCAGACCAGAAGAGTGGTGGTGGTGGTGGTGAGAAGAACCC",
            height=150,
            help="Enter the reverse DNA strand sequence"
        )
        
        vector_text = st.text_area(
            "Vector Sequence:",
            value="ATCCGGATATAGTTCCTCCTTTCAGCAAAAAACCCCTCAAGACCCGTTTAGAGGCCCCAAGGGGTTATGCTAGTTATTGCTCAGCGGCTCGAGCATATG",
            height=150,
            help="Enter the vector sequence"
        )
        
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            vector_name = st.text_input("Vector Name:", "pET-21a(+)")
        
        with col1_2:
            insert_name = st.text_input("Insert Name:", "Gene of Interest")
        
        enzyme_pair = st.selectbox(
            "Enzyme Pair:",
            list(enzyme_pairs.keys()),
            help="Select the restriction enzyme pair for cloning"
        )
        
        col1_3, col1_4 = st.columns(2)
        
        with col1_3:
            colorblind_mode = st.checkbox("Colorblind Mode")
        
        with col1_4:
            show_sequence = st.checkbox("Show Sequence Details", True)
        
        # Action buttons
        col1_5, col1_6 = st.columns(2)
        
        with col1_5:
            if st.button("🔬 Digest Vector", type="primary"):
                if vector_text:
                    with st.spinner("Digesting vector..."):
                        try:
                            # Simulate vector digestion
                            clean_vector = clean_dna_sequence(vector_text)
                            
                            digest_result = {
                                'original_sequence': clean_vector,
                                'size': len(clean_vector),
                                'gc_content': calculate_gc(clean_vector),
                                'cut_sites': 2,  # Assuming both enzymes cut
                                'linearized': True,
                                'enzyme_pair': enzyme_pair
                            }
                            
                            st.session_state.vector_digested = True
                            st.session_state.digested_vector = digest_result
                            
                            # Add to history
                            st.session_state.history.append({
                                'timestamp': datetime.now().strftime("%H:%M:%S"),
                                'action': f"Digested vector with {enzyme_pair}",
                                'tool': 'Ligation Check'
                            })
                            
                            create_status_message("✅ Vector digested successfully!", "success")
                        
                        except Exception as e:
                            logger.error(f"Error in vector digestion: {e}")
                            create_status_message(f"❌ Digestion error: {str(e)}", "error")
                else:
                    create_status_message("❌ Please enter a vector sequence", "error")
        
        with col1_6:
            if st.button("🔗 Check Ligation"):
                if not st.session_state.get('vector_digested', False):
                    create_status_message("⚠️ Please digest the vector first", "warning")
                else:
                    if forward_text and reverse_text:
                        with st.spinner("Checking ligation compatibility..."):
                            try:
                                # Clean sequences
                                forward_clean = clean_dna_sequence(forward_text)
                                reverse_clean = clean_dna_sequence(reverse_text)
                                
                                # Create combined insert
                                combined_insert = forward_clean + reverse_clean
                                
                                # Check for issues
                                issues = []
                                
                                # Check if insert length is multiple of 3 (for coding sequences)
                                if len(combined_insert) % 3 != 0:
                                    issues.append("Insert length is not a multiple of 3 (may cause frameshift)")
                                
                                # Check GC content compatibility
                                insert_gc = calculate_gc(combined_insert)
                                vector_gc = st.session_state.digested_vector.get('gc_content', 50)
                                
                                if abs(insert_gc - vector_gc) > 20:
                                    issues.append(f"GC content mismatch: Insert {insert_gc:.1f}% vs Vector {vector_gc:.1f}%")
                                
                                # Create ligated sequence (simplified)
                                vector_seq = st.session_state.digested_vector['original_sequence']
                                cut_position = len(vector_seq) // 2  # Simplified cut position
                                ligated_sequence = vector_seq[:cut_position] + combined_insert + vector_seq[cut_position:]
                                
                                ligation_result = {
                                    'compatible': len(issues) == 0,
                                    'issues': issues,
                                    'ligated_sequence': ligated_sequence,
                                    'insert_gc': insert_gc,
                                    'vector_gc': vector_gc,
                                    'insert_length': len(combined_insert),
                                    'final_length': len(ligated_sequence)
                                }
                                
                                st.session_state.ligation_result = ligation_result
                                
                                # Add to history
                                st.session_state.history.append({
                                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                                    'action': "Checked ligation compatibility",
                                    'tool': 'Ligation Check'
                                })
                                
                                if ligation_result['compatible']:
                                    create_status_message("✅ Ligation is compatible!", "success")
                                else:
                                    create_status_message("❌ Ligation compatibility issues detected", "error")
                            
                            except Exception as e:
                                logger.error(f"Error in ligation check: {e}")
                                create_status_message(f"❌ Ligation check error: {str(e)}", "error")
                    else:
                        create_status_message("❌ Please enter both forward and reverse sequences", "error")
    
    with col2:
        st.markdown("### Ligation Analysis")
        
        if st.session_state.get('vector_digested', False):
            create_status_message("✅ Vector Status: Digested", "success")
            
            # Show digested vector information
            digest_info = st.session_state.digested_vector
            if digest_info:
                st.markdown("#### Digestion Details")
                col2_1, col2_2, col2_3, col2_4 = st.columns(4)
                
                with col2_1:
                    create_metric_card("Vector Size", f"{digest_info.get('size', 0)} bp")
                
                with col2_2:
                    create_metric_card("Cut Sites", str(digest_info.get('cut_sites', 0)))
                
                with col2_3:
                    create_metric_card("GC Content", f"{digest_info.get('gc_content', 0):.1f}%")
                
                with col2_4:
                    create_metric_card("Status", "Linearized" if digest_info.get('linearized', False) else "Circular")
            
            # Show ligation results if available
            if 'ligation_result' in st.session_state:
                result = st.session_state.ligation_result
                
                st.markdown("#### Ligation Results")
                
                # Compatibility status
                if result['compatible']:
                    create_status_message("🎯 **Ligation Status: Compatible**", "success")
                    
                    # Show construct metrics
                    col2_5, col2_6, col2_7, col2_8 = st.columns(4)
                    
                    with col2_5:
                        create_metric_card("Insert Length", f"{result['insert_length']} bp")
                    
                    with col2_6:
                        create_metric_card("Final Length", f"{result['final_length']} bp")
                    
                    with col2_7:
                        create_metric_card("Insert GC%", f"{result['insert_gc']:.1f}%")
                    
                    with col2_8:
                        create_metric_card("Vector GC%", f"{result['vector_gc']:.1f}%")
                    
                    if show_sequence and result.get('ligated_sequence'):
                        st.markdown("#### Ligated Sequence")
                        with st.expander("🧬 View Sequence", expanded=False):
                            st.markdown(f'<div class="sequence-display">{result["ligated_sequence"]}</div>', 
                                      unsafe_allow_html=True)
                            
                            # Download option
                            fasta_content = f">{vector_name}_{insert_name}_ligated\n{result['ligated_sequence']}"
                            create_download_button(fasta_content, f"{vector_name}_{insert_name}.fasta", "Download Sequence")
                
                else:
                    create_status_message("❌ **Ligation Status: Incompatible**", "error")
                    
                    # Show compatibility issues
                    if result.get('issues'):
                        st.markdown("#### Compatibility Issues:")
                        for issue in result['issues']:
                            st.warning(f"⚠️ {issue}")
                    
                    # Suggestions for fixing issues
                    st.markdown("#### Suggestions:")
                    st.info("💡 Try adjusting enzyme pair or check sequence orientation")
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%); 
                        border-radius: 12px; border: 2px dashed #cbd5e1;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">✂️</div>
                <div style="color: #64748b; font-size: 1.1rem;">
                    Digest a vector to begin ligation analysis
                </div>
            </div>
            """, unsafe_allow_html=True)

# Primer Generator tab
def render_primer_generator():
    """Render the Primer Generator tab"""
    st.markdown("## 🎯 Primer Generator")
    st.markdown("Design PCR primers for DNA amplification.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Input Parameters")
        
        template_sequence = st.text_area(
            "Template Sequence:", 
            "ATGCTAGCTAGCTCGAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTACGTACGT", 
            height=150,
            help="Enter the template DNA sequence for primer design"
        )
        
        # Real-time validation
        if template_sequence:
            is_valid, clean_seq, warning = validate_dna_sequence(template_sequence)
            if not is_valid:
                create_status_message(f"⚠️ {warning}", "warning")
            elif warning:
                create_status_message(f"ℹ️ {warning}", "info")
            else:
                create_status_message(f"✅ Valid template sequence ({len(clean_seq)} bp)", "success")
        
        st.markdown("#### Target Region (Optional)")
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            start_pos = st.number_input("Start Position", 0, 1000, 0)
        with col1_2:
            end_pos = st.number_input("End Position", 0, 1000, 0)
        
        if end_pos == 0:
            end_pos = len(template_sequence) if template_sequence else 0
        
        st.markdown("#### Primer Parameters")
        col1_3, col1_4 = st.columns(2)
        with col1_3:
            min_length = st.slider("Min Length", 15, 25, 18)
        with col1_4:
            max_length = st.slider("Max Length", 25, 40, 30)
        
        col1_5, col1_6 = st.columns(2)
        with col1_5:
            target_tm = st.slider("Target Tm (°C)", 50.0, 70.0, 60.0)
        with col1_6:
            gc_range = st.slider("GC Content Range (%)", 40, 60, (45, 55))
        
        # Advanced options
        if st.session_state.user_preferences.get('show_advanced_options', False):
            with st.expander("🔧 Advanced Options"):
                add_restrictions = st.checkbox("Add Restriction Sites")
                if add_restrictions:
                    col1_7, col1_8 = st.columns(2)
                    with col1_7:
                        f_restriction = st.selectbox("Forward Primer Site", list(enzyme_linkers.keys()))
                    with col1_8:
                        r_restriction = st.selectbox("Reverse Primer Site", list(enzyme_linkers.keys()), index=1)
                else:
                    f_restriction = None
                    r_restriction = None
                
                avoid_secondary = st.checkbox("Avoid Secondary Structures", value=True)
                primer_conc = st.slider("Primer Concentration (nM)", 100, 1000, 500)
        else:
            add_restrictions = False
            f_restriction = None
            r_restriction = None
            avoid_secondary = True
            primer_conc = 500
        
        if st.button("🎯 Design Primers", type="primary", use_container_width=True):
            if template_sequence:
                with st.spinner("Designing primers..."):
                    try:
                        # Clean sequence
                        is_valid, clean_seq, warning = validate_dna_sequence(template_sequence)
                        
                        if is_valid:
                            # Get target region
                            if 0 <= start_pos < end_pos <= len(clean_seq):
                                target_region = (start_pos, end_pos)
                                template_region = clean_seq[start_pos:end_pos]
                            else:
                                target_region = (0, len(clean_seq))
                                template_region = clean_seq
                            
                            # Design primers
                            primers = []
                            
                            for length in range(min_length, max_length + 1):
                                forward = template_region[:length]
                                reverse = reverse_complement(template_region[-length:])
                                
                                # Add restriction sites if requested
                                if add_restrictions and f_restriction and r_restriction:
                                    forward = enzyme_linkers[f_restriction] + forward
                                    reverse = enzyme_linkers[r_restriction] + reverse
                                
                                forward_tm = calculate_tm_consensus(forward, primer_conc=primer_conc*1e-9)
                                reverse_tm = calculate_tm_consensus(reverse, primer_conc=primer_conc*1e-9)
                                
                                if forward_tm and reverse_tm:
                                    forward_gc = calculate_gc(forward)
                                    reverse_gc = calculate_gc(reverse)
                                    
                                    # Check if TMs are close to target and GC is in range
                                    if (abs(forward_tm - target_tm) < 5 and abs(reverse_tm - target_tm) < 5 and
                                        gc_range[0] <= forward_gc <= gc_range[1] and
                                        gc_range[0] <= reverse_gc <= gc_range[1]):
                                        
                                        primers.append({
                                            'forward_primer': forward,
                                            'reverse_primer': reverse,
                                            'forward_tm': forward_tm,
                                            'reverse_tm': reverse_tm,
                                            'forward_gc': forward_gc,
                                            'reverse_gc': reverse_gc,
                                            'length': length,
                                            'amplicon_size': end_pos - start_pos
                                        })
                            
                            # Sort by how close TMs are to target
                            primers.sort(key=lambda x: (abs(x['forward_tm'] - target_tm) + abs(x['reverse_tm'] - target_tm)))
                            
                            # Store results
                            if primers:
                                primer_results = {
                                    'primers': primers,
                                    'template': clean_seq,
                                    'target_region': target_region,
                                    'parameters': {
                                        'min_length': min_length,
                                        'max_length': max_length,
                                        'target_tm': target_tm,
                                        'gc_range': gc_range,
                                        'add_restrictions': add_restrictions,
                                        'f_restriction': f_restriction,
                                        'r_restriction': r_restriction
                                    }
                                }
                                
                                st.session_state.current_results = primer_results
                                
                                # Add to history
                                st.session_state.history.append({
                                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                                    'action': f"Designed {len(primers)} primer pairs for {len(clean_seq)} bp template",
                                    'tool': 'Primer Generator'
                                })
                                
                                create_status_message(f"✅ Generated {len(primers)} primer pairs", "success")
                            else:
                                create_status_message("❌ Could not generate suitable primers. Try adjusting parameters.", "error")
                        else:
                            create_status_message(f"❌ Invalid template sequence: {warning}", "error")
                    
                    except Exception as e:
                        logger.error(f"Error in primer design: {e}")
                        create_status_message(f"❌ Primer design error: {str(e)}", "error")
            else:
                create_status_message("❌ Please enter a template sequence", "error")
    
    with col2:
        st.markdown("### Primer Design Results")
        
        if ('current_results' in st.session_state and 
            'primers' in st.session_state.current_results):
            
            results = st.session_state.current_results
            primers = results['primers']
            
            # Allow the user to switch between different primer pairs
            if len(primers) > 1:
                selected_pair = st.selectbox(
                    "Select Primer Pair:",
                    range(len(primers)),
                    format_func=lambda x: f"Pair {x+1}: {primers[x]['length']} bp primers (Tm: {primers[x]['forward_tm']:.1f}°C / {primers[x]['reverse_tm']:.1f}°C)"
                )
                primer_pair = primers[selected_pair]
            else:
                primer_pair = primers[0]
            
            # Display primer information
            st.markdown("#### Selected Primer Pair")
            
            st.markdown("**Forward Primer:**")
            st.markdown(f'<div class="sequence-display">{primer_pair["forward_primer"]}</div>', 
                      unsafe_allow_html=True)
            
            st.markdown("**Reverse Primer:**")
            st.markdown(f'<div class="sequence-display">{primer_pair["reverse_primer"]}</div>', 
                      unsafe_allow_html=True)
            
            # Properties
            st.markdown("#### Primer Properties")
            
            col2_1, col2_2, col2_3, col2_4 = st.columns(4)
            
            with col2_1:
                create_metric_card("Forward Length", f"{len(primer_pair['forward_primer'])} bp")
            
            with col2_2:
                create_metric_card("Reverse Length", f"{len(primer_pair['reverse_primer'])} bp")
            
            with col2_3:
                create_metric_card("Forward Tm", f"{primer_pair['forward_tm']:.1f}°C")
            
            with col2_4:
                create_metric_card("Reverse Tm", f"{primer_pair['reverse_tm']:.1f}°C")
            
            # Additional properties
            col2_5, col2_6, col2_7, col2_8 = st.columns(4)
            
            with col2_5:
                create_metric_card("Forward GC%", f"{primer_pair['forward_gc']:.1f}%")
            
            with col2_6:
                create_metric_card("Reverse GC%", f"{primer_pair['reverse_gc']:.1f}%")
            
            with col2_7:
                create_metric_card("Amplicon Size", f"{primer_pair['amplicon_size']} bp")
            
            with col2_8:
                tm_diff = abs(primer_pair['forward_tm'] - primer_pair['reverse_tm'])
                create_metric_card("Tm Difference", f"{tm_diff:.1f}°C")
            
            # Primer details table
            st.markdown("#### Detailed Analysis")
            
            details_data = {
                'Property': ['Sequence', 'Length (bp)', 'Tm (°C)', 'GC Content (%)', 'MW (Da)'],
                'Forward Primer': [
                    primer_pair['forward_primer'],
                    len(primer_pair['forward_primer']),
                    f"{primer_pair['forward_tm']:.1f}",
                    f"{primer_pair['forward_gc']:.1f}",
                    len(primer_pair['forward_primer']) * 330  # Approximate MW
                ],
                'Reverse Primer': [
                    primer_pair['reverse_primer'],
                    len(primer_pair['reverse_primer']),
                    f"{primer_pair['reverse_tm']:.1f}",
                    f"{primer_pair['reverse_gc']:.1f}",
                    len(primer_pair['reverse_primer']) * 330  # Approximate MW
                ]
            }
            
            details_df = pd.DataFrame(details_data)
            st.dataframe(details_df, use_container_width=True, hide_index=True)
            
            # Download options
            st.markdown("#### Download Options")
            col2_9, col2_10 = st.columns(2)
            
            with col2_9:
                # FASTA format
                primer_fasta = f">Forward_Primer\n{primer_pair['forward_primer']}\n>Reverse_Primer\n{primer_pair['reverse_primer']}"
                create_download_button(primer_fasta, "primers.fasta", "Download FASTA")
            
            with col2_10:
                # Detailed report
                report_content = f"""G-Synth Primer Design Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

Template Sequence Length: {len(results['template'])} bp
Target Region: {results['target_region'][0]}-{results['target_region'][1]}
Amplicon Size: {primer_pair['amplicon_size']} bp

Forward Primer:
Sequence: {primer_pair['forward_primer']}
Length: {len(primer_pair['forward_primer'])} bp
Tm: {primer_pair['forward_tm']:.1f}°C
GC Content: {primer_pair['forward_gc']:.1f}%

Reverse Primer:
Sequence: {primer_pair['reverse_primer']}
Length: {len(primer_pair['reverse_primer'])} bp
Tm: {primer_pair['reverse_tm']:.1f}°C
GC Content: {primer_pair['reverse_gc']:.1f}%

Design Parameters:
- Target Tm: {results['parameters']['target_tm']}°C
- Length Range: {results['parameters']['min_length']}-{results['parameters']['max_length']} bp
- GC Range: {results['parameters']['gc_range'][0]}-{results['parameters']['gc_range'][1]}%
"""
                create_download_button(report_content, "primer_report.txt", "Download Report")
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%); 
                        border-radius: 12px; border: 2px dashed #cbd5e1;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
                <div style="color: #64748b; font-size: 1.1rem;">
                    Enter template sequence and design primers to see results
                </div>
            </div>
            """, unsafe_allow_html=True)

# Reverse Complement tab
def render_reverse_complement():
    """Render the Reverse Complement tab"""
    st.markdown("## 🔄 Reverse Complement")
    st.markdown("Generate reverse, complement, or reverse-complement of a DNA sequence.")
    
    # Input section
    dna_sequence = st.text_area(
        "DNA Sequence:",
        value="ATGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC",
        height=150,
        help="Enter the DNA sequence to process"
    )
    
    # Real-time validation
    if dna_sequence:
        is_valid, clean_seq, warning = validate_dna_sequence(dna_sequence)
        if not is_valid:
            create_status_message(f"⚠️ {warning}", "warning")
        elif warning:
            create_status_message(f"ℹ️ {warning}", "info")
        else:
            create_status_message(f"✅ Valid DNA sequence ({len(clean_seq)} bp)", "success")
    
    # Options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_original = st.checkbox("Show Original", True)
    with col2:
        highlight_differences = st.checkbox("Highlight Changes", True)
    with col3:
        case_sensitive = st.checkbox("Preserve Case", False)
    
    # Operation buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("↩️ Generate Reverse", type="primary", use_container_width=True):
            if dna_sequence:
                try:
                    is_valid, clean_seq, warning = validate_dna_sequence(dna_sequence)
                    if is_valid:
                        result = clean_seq[::-1]
                        
                        # Store result
                        reverse_result = {
                            'operation': 'Reverse',
                            'original': clean_seq,
                            'result': result,
                            'warning': warning
                        }
                        st.session_state.current_results = reverse_result
                        
                        # Add to history
                        st.session_state.history.append({
                            'timestamp': datetime.now().strftime("%H:%M:%S"),
                            'action': f"Generated reverse of {len(clean_seq)} bp sequence",
                            'tool': 'Reverse Complement'
                        })
                        
                        create_status_message("✅ Reverse sequence generated", "success")
                    else:
                        create_status_message(f"❌ {warning}", "error")
                
                except Exception as e:
                    logger.error(f"Error generating reverse: {e}")
                    create_status_message(f"❌ Error: {str(e)}", "error")
    
    with col2:
        if st.button("🔄 Generate Complement", type="primary", use_container_width=True):
            if dna_sequence:
                try:
                    is_valid, clean_seq, warning = validate_dna_sequence(dna_sequence)
                    if is_valid:
                        complement = str.maketrans('ATCGatcg', 'TAGCtagc')
                        result = clean_seq.translate(complement)
                        
                        # Store result
                        complement_result = {
                            'operation': 'Complement',
                            'original': clean_seq,
                            'result': result,
                            'warning': warning
                        }
                        st.session_state.current_results = complement_result
                        
                        # Add to history
                        st.session_state.history.append({
                            'timestamp': datetime.now().strftime("%H:%M:%S"),
                            'action': f"Generated complement of {len(clean_seq)} bp sequence",
                            'tool': 'Reverse Complement'
                        })
                        
                        create_status_message("✅ Complement sequence generated", "success")
                    else:
                        create_status_message(f"❌ {warning}", "error")
                
                except Exception as e:
                    logger.error(f"Error generating complement: {e}")
                    create_status_message(f"❌ Error: {str(e)}", "error")
    
    with col3:
        if st.button("🔄↩️ Generate Reverse Complement", type="primary", use_container_width=True):
            if dna_sequence:
                try:
                    is_valid, clean_seq, warning = validate_dna_sequence(dna_sequence)
                    if is_valid:
                        result = reverse_complement(clean_seq)
                        
                        # Store result
                        rc_result = {
                            'operation': 'Reverse Complement',
                            'original': clean_seq,
                            'result': result,
                            'warning': warning
                        }
                        st.session_state.current_results = rc_result
                        
                        # Add to history
                        st.session_state.history.append({
                            'timestamp': datetime.now().strftime("%H:%M:%S"),
                            'action': f"Generated reverse complement of {len(clean_seq)} bp sequence",
                            'tool': 'Reverse Complement'
                        })
                        
                        create_status_message("✅ Reverse complement generated", "success")
                    else:
                        create_status_message(f"❌ {warning}", "error")
                
                except Exception as e:
                    logger.error(f"Error generating reverse complement: {e}")
                    create_status_message(f"❌ Error: {str(e)}", "error")
    
    # Results section
    if 'current_results' in st.session_state and st.session_state.current_results:
        result = st.session_state.current_results
        
        st.markdown("---")
        st.markdown(f"### {result['operation']} Results")
        
        if result.get('warning'):
            create_status_message(f"ℹ️ {result['warning']}", "info")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if show_original:
                st.markdown("#### Original Sequence")
                st.markdown(f'<div class="sequence-display">{result["original"]}</div>', 
                          unsafe_allow_html=True)
                
                # Properties
                original_props = {
                    'Length': f"{len(result['original'])} bp",
                    'GC Content': f"{calculate_gc(result['original']):.1f}%",
                    'Tm': f"{calculate_tm_consensus(result['original']) or 'N/A'}"
                }
                
                for prop, value in original_props.items():
                    st.metric(prop, value)
        
        with col2:
            st.markdown(f"#### {result['operation']} Sequence")
            st.markdown(f'<div class="sequence-display">{result["result"]}</div>', 
                      unsafe_allow_html=True)
            
            # Properties
            result_props = {
                'Length': f"{len(result['result'])} bp",
                'GC Content': f"{calculate_gc(result['result']):.1f}%",
                'Tm': f"{calculate_tm_consensus(result['result']) or 'N/A'}"
            }
            
            for prop, value in result_props.items():
                st.metric(prop, value)
        
        # Analysis section
        st.markdown("#### Sequence Analysis")
        
        if result['operation'] == 'Reverse Complement':
            st.info("💡 The reverse complement is the sequence that would hybridize to the original sequence in antiparallel orientation.")
        elif result['operation'] == 'Complement':
            st.info("💡 The complement shows the base-pairing partner for each nucleotide.")
        else:
            st.info("💡 The reverse sequence shows the original sequence read backwards.")
        
        # Download options
        st.markdown("#### Download Options")
        col1, col2 = st.columns(2)
        
        with col1:
            # FASTA format
            fasta_content = f">Original_Sequence\n{result['original']}\n>{result['operation'].replace(' ', '_')}_Sequence\n{result['result']}"
            create_download_button(fasta_content, f"{result['operation'].lower().replace(' ', '_')}.fasta", "Download FASTA")
        
        with col2:
            # Report format
            report_content = f"""G-Synth {result['operation']} Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

Operation: {result['operation']}

Original Sequence:
{result['original']}

{result['operation']} Sequence:
{result['result']}

Properties Comparison:
                     Original    {result['operation']}
Length (bp):         {len(result['original']):<10} {len(result['result'])}
GC Content (%):      {calculate_gc(result['original']):<10.1f} {calculate_gc(result['result']):.1f}
"""
            create_download_button(report_content, f"{result['operation'].lower().replace(' ', '_')}_report.txt", "Download Report")

# Help & Guide tab
def render_help_tab():
    """Render the Help & Guide tab"""
    st.markdown("## ❓ Help & Guide")
    st.markdown("Comprehensive documentation and tutorials for G-Synth toolkit.")
    
    # Create tabs for different help sections
    help_tab1, help_tab2, help_tab3, help_tab4 = st.tabs([
        "📖 Quick Start", "🔬 Tool Guides", "🧬 Reference", "❓ FAQ"
    ])
    
    with help_tab1:
        st.markdown("### 🚀 Quick Start Guide")
        
        st.markdown("""
        Welcome to G-Synth! This toolkit provides comprehensive solutions for genetic engineering and molecular cloning.
        
        #### Getting Started:
        
        1. **Select a Tool**: Use the sidebar to navigate between different tools
        2. **Enter Your Data**: Input DNA/protein sequences in the provided text areas
        3. **Configure Parameters**: Adjust settings based on your experimental needs
        4. **Run Analysis**: Click the process button to perform calculations
        5. **Download Results**: Save your results in various formats
        
        #### Common Workflows:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔬 For Cloning:**
            1. Use "Small Sequence Design" to design your insert
            2. Use "Primer Generator" to design PCR primers
            3. Use "Ligation Check" to verify compatibility
            """)
            
            st.markdown("""
            **📏 For Long Sequences:**
            1. Use "Extended Synthesis" to fragment sequences
            2. Use "Hybridization Simulation" for assembly
            3. Use "Codon Optimization" for expression
            """)
        
        with col2:
            st.markdown("""
            **⚡ For Expression:**
            1. Use "Codon Optimization" for your host
            2. Use "Translation" to verify protein sequence
            3. Use "Small Sequence Design" to add tags
            """)
            
            st.markdown("""
            **🔄 For Analysis:**
            1. Use "Translation" for sequence analysis
            2. Use "Reverse Complement" for primer design
            3. Use "Hybridization" for oligonucleotide design
            """)
    
    with help_tab2:
        st.markdown("### 🔬 Tool-Specific Guides")
        
        tool_guides = {
            "Small Sequence Design": {
                "description": "Design short DNA sequences with enzyme sites",
                "inputs": ["DNA sequence", "Enzyme pair", "Sequence type"],
                "outputs": ["Forward/reverse sequences", "Properties", "Analysis"],
                "tips": [
                    "Use coding sequences for protein expression",
                    "Add cleavage sites for protein purification",
                    "Check GC content for synthesis feasibility"
                ]
            },
            "Translation & Reverse Translation": {
                "description": "Convert between DNA and protein sequences",
                "inputs": ["DNA or protein sequence", "Reading frame", "Target organism"],
                "outputs": ["Translated sequence", "Optimization metrics", "Verification"],
                "tips": [
                    "Select correct reading frame for translation",
                    "Use organism-specific codon optimization",
                    "Verify back-translation matches original"
                ]
            },
            "Codon Optimization": {
                "description": "Optimize sequences for expression hosts",
                "inputs": ["DNA/protein sequence", "Target organism", "Parameters"],
                "outputs": ["Optimized sequence", "Change statistics", "Analysis"],
                "tips": [
                    "E. coli optimization for bacterial expression",
                    "Human optimization for mammalian cells",
                    "Balance GC content and codon usage"
                ]
            }
        }
        
        selected_guide = st.selectbox("Select Tool:", list(tool_guides.keys()))
        guide = tool_guides[selected_guide]
        
        create_feature_card("🔬", selected_guide, guide["description"])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Inputs:**")
            for inp in guide["inputs"]:
                st.write(f"• {inp}")
        
        with col2:
            st.markdown("**Outputs:**")
            for out in guide["outputs"]:
                st.write(f"• {out}")
        
        with col3:
            st.markdown("**Tips:**")
            for tip in guide["tips"]:
                st.write(f"💡 {tip}")
    
    with help_tab3:
        st.markdown("### 🧬 Molecular Biology Reference")
        
        ref_tab1, ref_tab2, ref_tab3 = st.tabs(["Genetic Code", "Restriction Enzymes", "Calculations"])
        
        with ref_tab1:
            st.markdown("#### Standard Genetic Code")
            
            # Create genetic code table
            genetic_code_data = []
            for codon, aa in genetic_code.items():
                aa_name = {
                    'A': 'Alanine', 'C': 'Cysteine', 'D': 'Aspartic acid', 'E': 'Glutamic acid',
                    'F': 'Phenylalanine', 'G': 'Glycine', 'H': 'Histidine', 'I': 'Isoleucine',
                    'K': 'Lysine', 'L': 'Leucine', 'M': 'Methionine', 'N': 'Asparagine',
                    'P': 'Proline', 'Q': 'Glutamine', 'R': 'Arginine', 'S': 'Serine',
                    'T': 'Threonine', 'V': 'Valine', 'W': 'Tryptophan', 'Y': 'Tyrosine',
                    '*': 'Stop'
                }.get(aa, 'Unknown')
                
                genetic_code_data.append({
                    "Codon": codon,
                    "Amino Acid": aa,
                    "Name": aa_name
                })
            
            genetic_df = pd.DataFrame(genetic_code_data)
            st.dataframe(genetic_df, use_container_width=True, hide_index=True)
        
        with ref_tab2:
            st.markdown("#### Common Restriction Enzymes")
            
            enzyme_data = []
            for enzyme, site in enzyme_linkers.items():
                enzyme_data.append({
                    "Enzyme": enzyme,
                    "Recognition Site": site,
                    "Length": len(site),
                    "Type": "Type II",
                    "Overhang": "5'" if enzyme in ["EcoRI", "BamHI"] else "3'" if enzyme in ["PstI"] else "Blunt"
                })
            
            enzyme_df = pd.DataFrame(enzyme_data)
            st.dataframe(enzyme_df, use_container_width=True, hide_index=True)
        
        with ref_tab3:
            st.markdown("#### Useful Calculations")
            
            st.markdown("""
            **GC Content:**
            ```
            GC% = (G + C) / (A + T + G + C) × 100
            ```
            
            **Melting Temperature (Wallace Rule):**
            ```
            Tm = 2(A + T) + 4(G + C) - 7°C  (for sequences < 14 bp)
            ```
            
            **Nearest-Neighbor Tm:**
            ```
            Tm = ΔH / (ΔS + R × ln(C)) - 273.15 + salt_correction
            ```
            
            **DNA Concentration:**
            ```
            μg/ml = A260 × dilution × 50 μg/ml
            ```
            
            **Molecular Weight:**
            ```
            MW (Da) = (A×331 + T×322 + G×347 + C×306) - 61
            ```
            
            **Copy Number:**
            ```
            copies = (concentration × 6.022×10²³) / MW
            ```
            """)
    
    with help_tab4:
        st.markdown("### ❓ Frequently Asked Questions")
        
        faqs = [
            {
                "question": "What file formats can I use for input?",
                "answer": "You can paste sequences directly into text areas. The app accepts plain text DNA sequences (A, T, C, G) and protein sequences (single letter amino acid codes). Files can be copied and pasted from FASTA, GenBank, or plain text formats."
            },
            {
                "question": "How accurate are the Tm calculations?",
                "answer": "G-Synth uses nearest-neighbor thermodynamics for accurate Tm predictions. For primers <30bp, accuracy is typically ±2°C under standard conditions (50mM Na+, 500nM primer). For longer sequences, the accuracy may vary ±5°C."
            },
            {
                "question": "Can I optimize sequences for custom organisms?",
                "answer": "Currently, optimization is available for E. coli, S. cerevisiae, H. sapiens, and CHO cells. The codon usage tables are based on the most recent genomic data. Custom organisms can be added by modifying the CODON_USAGE_TABLES dictionary."
            },
            {
                "question": "What's the maximum sequence length?",
                "answer": "There's no hard limit, but very long sequences (>10kb) may take longer to process and could cause browser memory issues. For sequences >5kb, we recommend using the Extended Synthesis tool for fragmentation."
            },
            {
                "question": "How do I report bugs or request features?",
                "answer": "Please contact the development team with your browser version and steps to reproduce any issues. Include the sequence data and parameters used when reporting bugs."
            },
            {
                "question": "Are my sequences stored or transmitted?",
                "answer": "All processing is done locally in your browser. Sequences are not transmitted to servers or stored permanently. Session data is only kept in browser memory and is cleared when you close the application."
            }
        ]
        
        for i, faq in enumerate(faqs):
            with st.expander(f"Q{i+1}: {faq['question']}"):
                st.write(faq['answer'])
        
        st.markdown("---")
        st.markdown("#### 📧 Contact & Support")
        st.info("""
        **Need more help?**
        
        - 📧 Email: support@g-synth.dev
        - 📖 Documentation: Complete user manual with examples
        - 💬 Community: Discussion forum for users
        - 🎓 Tutorials: Step-by-step video guides
        
        **Developer Information:**
        G-Synth was developed by Dr. Mohamed Merzoug for the scientific community.
        Version 2025.5.0 - Enhanced Interactive Web Interface
        """)

# Main application function
def main():
    """Main application function with error handling"""
    try:
        # Render header
        render_header()
        
        # Render sidebar and get selected tab
        selected_tab = render_sidebar()
        
        # Render the selected tab based on sidebar selection
        if selected_tab == "Home":
            render_home_tab()
        elif selected_tab == "Small Sequence Design":
            render_small_sequence_design_tab()
        elif selected_tab == "Translation & Reverse Translation":
            render_translation_tab()
        elif selected_tab == "Codon Optimization":
            render_codon_optimization_tab()
        elif selected_tab == "Extended Synthesis":
            render_extended_synthesis_tab()
        elif selected_tab == "Hybridization Simulation":
            render_hybridization_tab()
        elif selected_tab == "Ligation Check":
            render_ligation_check_tab()
        elif selected_tab == "Primer Generator":
            render_primer_generator()
        elif selected_tab == "Reverse Complement":
            render_reverse_complement()
        elif selected_tab == "Help & Guide":
            render_help_tab()
        else:
            st.error(f"Unknown tab: {selected_tab}")
            st.info("Please select a valid tool from the sidebar.")
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An unexpected error occurred. Please refresh the page and try again.")
        st.exception(e)

# Application entry point
if __name__ == "__main__":
    main()