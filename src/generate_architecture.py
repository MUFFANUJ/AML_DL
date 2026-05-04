"""
Publication-ready architecture diagram using Graphviz.
Three-column layout: Phase 1 (ML), Phase 2 (DL), Phase 3 (Hybrid).
Output: assets/Architecture.png
"""

import os
from graphviz import Digraph

os.makedirs('assets', exist_ok=True)

dot = Digraph(name='Architecture', format='png')
dot.attr(
    rankdir='TB',
    splines='polyline',
    nodesep='0.90',
    ranksep='0.75',
    fontname='Helvetica',
    bgcolor='#FAFAFA',
    dpi='200',
    pad='0.7',
    newrank='true',
)
dot.attr('node', shape='box', style='filled,rounded',
         fontname='Helvetica', fontsize='11',
         margin='0.28,0.16', penwidth='1.8')
dot.attr('edge', fontname='Helvetica', fontsize='9',
         penwidth='1.5', arrowsize='0.85', color='#455A64')

# ── colour palette ─────────────────────────────────────────────────────────────
ML_F  = '#FFF8E1'; ML_C  = '#BF360C'; ML_FC = '#BF360C'   # orange  – ML
DL_F  = '#E3F2FD'; DL_C  = '#0D47A1'; DL_FC = '#0D47A1'   # blue    – DL
HY_F  = '#E8F5E9'; HY_C  = '#1B5E20'; HY_FC = '#1B5E20'   # green   – Hybrid
FU_F  = '#EDE7F6'; FU_C  = '#4527A0'; FU_FC = '#4527A0'   # purple  – Fusion
ERR_F = '#FFEBEE'; ERR_C = '#C62828'; ERR_FC = '#C62828'  # red     – Error
OK_F  = '#E8F5E9'; OK_C  = '#2E7D32'; OK_FC = '#2E7D32'   # green   – Correct

def ml(**kw):
    return dict(fillcolor=ML_F, color=ML_C, fontcolor=ML_FC, **kw)

def dl(**kw):
    return dict(fillcolor=DL_F, color=DL_C, fontcolor=DL_FC, **kw)

def hy(**kw):
    return dict(fillcolor=HY_F, color=HY_C, fontcolor=HY_FC, **kw)

def fu(**kw):
    return dict(fillcolor=FU_F, color=FU_C, fontcolor=FU_FC, **kw)

def err(**kw):
    return dict(fillcolor=ERR_F, color=ERR_C, fontcolor=ERR_FC, **kw)

def ok(**kw):
    return dict(fillcolor=OK_F, color=OK_C, fontcolor=OK_FC, **kw)

KEY = dict(penwidth='2.8', fontname='Helvetica-Bold')  # highlight key components

# ══════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════
dot.node('title',
         'Dynamic Trend & Event Detection Architecture\n(Hybrid ML + DL Pipeline)',
         shape='box', style='filled,rounded',
         fillcolor='#37474F', color='#263238', fontcolor='white',
         fontname='Helvetica-Bold', fontsize='15',
         penwidth='2.5', width='7.5', margin='0.35,0.22')

# ══════════════════════════════════════════════════════════════════════
# SHARED HEADER — data source
# ══════════════════════════════════════════════════════════════════════
dot.node('hdr',
         'HuffPost News Corpus\n210,000 articles  ·  2012–2022\nBalanced sample: 11,000 docs (1,000 / year)',
         **fu(fontname='Helvetica-Bold', fontsize='12', penwidth='2.2', width='5.5'))

# ══════════════════════════════════════════════════════════════════════
# PHASE 1 — ML  (LDA + Temporal Features)
# ══════════════════════════════════════════════════════════════════════
with dot.subgraph(name='cluster_p1') as c:
    c.attr(label='PHASE 1  —  ML  (LDA + Temporal Features)',
           labelloc='t', style='filled,rounded',
           fillcolor='#FFFDE7', color='#F57F17', fontcolor='#E65100',
           fontsize='12', fontname='Helvetica-Bold', penwidth='2.0', margin='18')

    c.node('a1', 'Text Cleaning\n· Lowercase  · Stopword removal\n· Regex filter  [^a-z\\s]',
           **ml())
    c.node('a2', 'TF-IDF Vectoriser\n· Vocab = 5,000  · max_df = 0.95\nOutput:  (N × 5,000)',
           **ml())
    c.node('a3', 'Latent Dirichlet Allocation  (LDA)\n· K = 10 topics  · 20 EM iterations\nOutput:  (N × 10) topic distribution',
           **ml(penwidth='2.2', fontname='Helvetica-Bold'))
    c.node('a4', 'Temporal Feature Engineering\n· Recency (F1)  · Category Velocity (F2)\n· Text Richness (F3)\nOutput:  (N × 3)',
           **ml())
    c.node('a5', 'Semantic Velocity  V(k,t)\n· Topic growth rate per time window\n· Tracks LDA topic surge',
           **ml())
    c.node('a6', '⚠  ML Failure — COVID peak: 2014  [WRONG]\nBoW conflates ACA legal (2014) with COVID (2020)\ncos(BoW_ACA, BoW_COVID) ≈ 0.80',
           **err(penwidth='2.2', fontname='Helvetica-Bold'))

    c.edges([('a1','a2'), ('a2','a3'), ('a3','a4'), ('a4','a5'), ('a5','a6')])

# ══════════════════════════════════════════════════════════════════════
# PHASE 2 — DL  (SBERT + TPI + UMAP + HDBSCAN)
# ══════════════════════════════════════════════════════════════════════
with dot.subgraph(name='cluster_p2') as c:
    c.attr(label='PHASE 2  —  DL  (SBERT + TPI + UMAP + HDBSCAN)',
           labelloc='t', style='filled,rounded',
           fillcolor='#E1F5FE', color='#01579B', fontcolor='#0D47A1',
           fontsize='12', fontname='Helvetica-Bold', penwidth='2.0', margin='18')

    c.node('b1', 'Text Cleaning  (DL variant)\n· Retains digits: "covid 19", "2020"\n· Regex filter  [^a-z0-9\\s]',
           **dl())
    # KEY: Sentence-BERT
    c.node('b2', 'Sentence-BERT  (all-MiniLM-L6-v2)\n· 22M params  · Siamese mean-pool\nOutput:  e_d  ∈  ℝ³⁸⁴',
           **dl(**KEY))
    # KEY: TPI
    c.node('b3', 'Temporal Positional Injection  [TPI]\n· Sinusoidal encoding:  PE(t)  ∈  ℝ³²\n· Fusion:  ẽ_d = [ e_d ∥ PE(t) ]  →  ℝ⁴¹⁶',
           **fu(**KEY))
    c.node('b4', 'TruncatedSVD  (pre-reduction)\n· Compact dense representation\n· (N × 416)  →  (N × 50)',
           **dl())
    c.node('b5', 'UMAP  Dimensionality Reduction\n· n_neighbors = 15  · cosine metric\n· (N × 50)  →  (N × 5)',
           **dl())
    c.node('b6', 'HDBSCAN  Density Clustering\n· min_cluster = 50  · min_samples = 10\n· K = 14 topics auto-discovered',
           **dl(penwidth='2.2', fontname='Helvetica-Bold'))
    c.node('b7', 'Class-based TF-IDF  (c-TF-IDF)\n· Cluster-level term weighting\n· ngram (1,2)  →  K topic vectors',
           **dl())
    c.node('b8', '✓  DL Result — COVID peak: 2020  [CORRECT]\ncross-cos(ACA, COVID) = 0.079  ·  TPI gap = 0.184\nStandalone precision = 83%',
           **ok(penwidth='2.2', fontname='Helvetica-Bold'))

    c.edges([('b1','b2'), ('b2','b3'), ('b3','b4'), ('b4','b5'),
             ('b5','b6'), ('b6','b7'), ('b7','b8')])

# ══════════════════════════════════════════════════════════════════════
# PHASE 3 — HYBRID  (BERTrend + GBM + GDELT)
# ══════════════════════════════════════════════════════════════════════
with dot.subgraph(name='cluster_p3') as c:
    c.attr(label='PHASE 3  —  HYBRID  (BERTrend + GBM + GDELT)',
           labelloc='t', style='filled,rounded',
           fillcolor='#F1F8E9', color='#33691E', fontcolor='#1B5E20',
           fontsize='12', fontname='Helvetica-Bold', penwidth='2.0', margin='18')

    c.node('c1', 'BERTrend Signal Classifier\n· Trend score P(k,t) per topic\n· NOISE | WEAK | STRONG | EMERGING',
           **hy())
    # KEY: GBM (Hybrid Fusion)
    c.node('c2', 'GBM Classifier  [Hybrid Fusion:  ML ⊕ DL]\n· ML inputs:  F1 Recency,  F2 Velocity,  F3 Richness\n· DL inputs:  coherence, purity, log-size, velocity\n· 100 trees  ·  depth = 3  ·  isotonic calibration',
           **fu(**KEY))
    c.node('c3', 'GDELT DOC 2.0  External Verification\n· Query: top-2 c-TF-IDF terms per topic\n· ±6-month corroboration window',
           **hy())
    c.node('c4', '★  5 Real Events Confirmed  (100% LOO Precision)\nCOVID-19  ·  Vaccination  ·  Ukraine  ·  Trump  ·  Stormy Daniels\nBERTopic alone = 83%   →   +GBM = 100%  (whole > parts)',
           **ok(penwidth='2.2', fontname='Helvetica-Bold'))
    c.node('c5', 'Flask REST API  :8000  +  React UI  (7 pages)\n· /api/predict  ·  /api/events  ·  /api/summary\n· Live classifier  ·  Signal evolution dashboard',
           **hy())

    c.edges([('c1','c2'), ('c2','c3'), ('c3','c4'), ('c4','c5')])

# ══════════════════════════════════════════════════════════════════════
# SHARED FOOTER
# ══════════════════════════════════════════════════════════════════════
dot.node('ftr',
         'Live Prediction  ·  Signal Evolution Dashboard  ·  Event Timeline  ·  Token Interpretability',
         **fu(fontname='Helvetica-Bold', fontsize='11', penwidth='2.0', width='6.5'))

# ── Diagnostic Ablation (bottom-right, visually separate) ─────────────────────
dot.node('abl',
         'Diagnostic Ablation Study\n'
         '[1]  − SBERT   →  cross-cos +912%,  COVID peak −6 yr\n'
         '[2]  − TPI     →  temporal gap  0.184  →  0.157  (−14.7%)\n'
         '[3]  − GBM     →  precision  100%  →  83%  (+1 false positive)\n'
         '[4]  Full Hybrid  →  100% LOO precision  (whole > sum of parts)',
         shape='note', style='filled',
         fontname='Courier', fontsize='10',
         fillcolor='#ECEFF1', color='#607D8B', fontcolor='#263238', penwidth='1.2')

# ══════════════════════════════════════════════════════════════════════
# TOP-LEVEL CONNECTIONS
# ══════════════════════════════════════════════════════════════════════
dot.edge('title', 'hdr', color='#37474F', penwidth='2.2', style='bold')
dot.edge('hdr', 'a1', color=FU_C, penwidth='2.2', style='bold')
dot.edge('hdr', 'b1', color=FU_C, penwidth='2.2', style='bold')

# ── Cross-cluster feature flows ────────────────────────────────────────────────
dot.edge('a4', 'c2',
         label='Temporal Features (ML)\nF1 · F2 · F3',
         color=ML_C, style='dashed', penwidth='2.0', constraint='false')
dot.edge('b6', 'c1',
         label='DL Features\n(velocity, signal, nz)',
         color=DL_C, style='dashed', penwidth='2.0', constraint='false')
dot.edge('b7', 'c2',
         label='DL Features\n(coherence, purity)',
         color=DL_C, style='dashed', penwidth='2.0', constraint='false')

# ── Footer & ablation connections ─────────────────────────────────────────────
dot.edge('c5', 'ftr', color=FU_C, penwidth='2.2', style='bold')
dot.edge('abl', 'c2', style='dotted', color='#90A4AE',
         arrowhead='none', constraint='false', penwidth='1.0')

# ══════════════════════════════════════════════════════════════════════
# RANK CONSTRAINTS — enforce horizontal alignment across phases
# ══════════════════════════════════════════════════════════════════════
with dot.subgraph() as s:
    s.attr(rank='same'); s.node('title')
with dot.subgraph() as s:
    s.attr(rank='same'); s.node('hdr')
with dot.subgraph() as s:
    s.attr(rank='same'); s.node('a1'); s.node('b1'); s.node('c1')
with dot.subgraph() as s:
    s.attr(rank='same'); s.node('a3'); s.node('b3')
with dot.subgraph() as s:
    s.attr(rank='same'); s.node('a6'); s.node('b8'); s.node('c5')
with dot.subgraph() as s:
    s.attr(rank='same'); s.node('ftr'); s.node('abl')

dot.render('assets/Architecture', cleanup=True)
print('Saved: assets/Architecture.png')
