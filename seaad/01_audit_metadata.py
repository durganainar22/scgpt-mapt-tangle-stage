#!/usr/bin/env python
"""
v3 Step 1: donor-level audit of SEA-AD MTG snRNA-seq, from the per-cell metadata CSV only
(no expression data needed yet).

Answers, before any modelling:
  - how many donors, and how many per Braak stage / 3-group Braak target
  - is Braak confounded with cognitive status, ADNC, sex, age, APOE, chemistry / method
  - how many neurons per donor (can every donor contribute to a neuron model?)

Input:   data/SEAAD_MTG_RNAseq_final-nuclei_metadata.2026-06-22.csv  (see seaad/00_download.sh)
Outputs: seaad/results/01_donors.csv, seaad/results/01_audit.txt

Usage:   python seaad/01_audit_metadata.py
"""
import os
import sys

import pandas as pd

PROJECT_DIR = os.environ.get('SCGPT_PROJECT_DIR',
                             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_DIR)
META = 'data/SEAAD_MTG_RNAseq_final-nuclei_metadata.2026-06-22.csv'
OUT = 'seaad/results'
os.makedirs(OUT, exist_ok=True)

DONOR_COLS = ['Neurotypical reference', 'Sex', 'Age at Death', 'PMI', 'RIN', 'Braak', 'Thal',
              'CERAD score', 'Overall AD neuropathological Change', 'Cognitive Status',
              'APOE Genotype', 'Severely Affected Donor']
CELL_COLS = ['Donor ID', 'Class', 'Subclass', 'method', 'library_prep', 'Used in analysis']

# Braak -> 3 groups (the pre-registered v3 target). Anything not listed is reported, not guessed.
BRAAK_GROUP = {'Braak 0': '0-II', 'Braak I': '0-II', 'Braak II': '0-II',
               'Braak III': 'III-IV', 'Braak IV': 'III-IV',
               'Braak V': 'V-VI', 'Braak VI': 'V-VI'}

log = open(f'{OUT}/01_audit.txt', 'w', encoding='utf-8')


def say(*a):
    print(*a)
    print(*a, file=log)


def section(title, obj):
    say(f'\n=== {title} ===')
    say(obj.to_string() if hasattr(obj, 'to_string') else obj)


cells = pd.read_csv(META, usecols=CELL_COLS + DONOR_COLS, low_memory=False)
say(f'Cells in metadata: {len(cells):,}')
section('Used in analysis', cells['Used in analysis'].value_counts(dropna=False))
section('Class', cells['Class'].value_counts(dropna=False))

# Donor-level attributes must be constant within a donor - check instead of assuming
nuniq = cells.groupby('Donor ID')[DONOR_COLS].nunique(dropna=False).max()
bad = nuniq[nuniq > 1]
if len(bad):
    say(f'\nWARNING: donor-level columns that vary within a donor: {bad.to_dict()}')

donors = cells.groupby('Donor ID')[DONOR_COLS].first()
is_neuron = cells['Class'].astype(str).str.startswith('Neuronal')
donors['n_cells'] = cells.groupby('Donor ID').size()
donors['n_neurons'] = cells[is_neuron].groupby('Donor ID').size().reindex(donors.index, fill_value=0)
donors['n_neurons_used'] = (cells[is_neuron & cells['Used in analysis'].astype(bool)]
                            .groupby('Donor ID').size().reindex(donors.index, fill_value=0))
donors['methods'] = cells.groupby('Donor ID')['method'].agg(lambda s: '|'.join(sorted(s.astype(str).unique())))
donors['n_library_preps'] = cells.groupby('Donor ID')['library_prep'].nunique()
donors['braak_group'] = donors['Braak'].map(BRAAK_GROUP)

unmapped = donors.loc[donors['braak_group'].isna(), 'Braak'].value_counts(dropna=False)
if len(unmapped):
    say(f'\nNOTE: Braak values not mapped to a group (excluded from the target): {unmapped.to_dict()}')

donors.to_csv(f'{OUT}/01_donors.csv')
say(f'\nDonors: {len(donors)}  (neurotypical reference: '
    f'{int(donors["Neurotypical reference"].astype(str).eq("True").sum())})')

section('Donors per Braak stage', donors['Braak'].value_counts(dropna=False).sort_index())
section('Donors per 3-group target', donors['braak_group'].value_counts(dropna=False).sort_index())
section('Braak group x Neurotypical reference',
        pd.crosstab(donors['braak_group'], donors['Neurotypical reference'], dropna=False))

# Confounding checks: the v2 lesson was stage == diagnosis. Is that true here too?
for col in ['Cognitive Status', 'Overall AD neuropathological Change', 'Sex', 'APOE Genotype',
            'Thal', 'CERAD score', 'Severely Affected Donor', 'methods']:
    section(f'Braak group x {col} (donors)', pd.crosstab(donors['braak_group'], donors[col], dropna=False))

num = donors[['Age at Death', 'PMI', 'RIN', 'n_neurons_used', 'n_library_preps']].apply(pd.to_numeric, errors='coerce')
section('Numeric covariates by Braak group (median [min-max])',
        num.groupby(donors['braak_group']).agg(lambda s: f'{s.median():.1f} [{s.min():.1f}-{s.max():.1f}]'))

section('Neurons (used in analysis) per donor', donors['n_neurons_used'].describe().round(0))
section('Neuron subclasses (used in analysis)',
        cells[is_neuron & cells['Used in analysis'].astype(bool)]['Subclass'].value_counts())

say(f'\nSaved {OUT}/01_donors.csv and {OUT}/01_audit.txt')
log.close()
sys.exit(0)
