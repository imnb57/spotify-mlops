"""
Spotify Dataset - Report Visualizations
Clean white background, colorful lines, diverse chart types
Designed for inclusion in an academic report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
import os

warnings.filterwarnings('ignore')

# Clean academic style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.color': '#555555',
    'ytick.color': '#555555',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'font.family': 'sans-serif',
    'grid.color': '#E0E0E0',
    'grid.linewidth': 0.6,
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
})

# Color palette
C_BLUE = '#2196F3'
C_RED = '#E53935'
C_GREEN = '#43A047'
C_ORANGE = '#FB8C00'
C_PURPLE = '#8E24AA'
C_TEAL = '#00897B'
C_PINK = '#D81B60'
C_AMBER = '#FFB300'
C_INDIGO = '#3949AB'
C_CYAN = '#00ACC1'
C_LIME = '#7CB342'
C_BROWN = '#6D4C41'
PALETTE_WARM = [C_RED, C_ORANGE, C_AMBER, C_PINK, C_PURPLE]
PALETTE_COOL = [C_BLUE, C_TEAL, C_GREEN, C_CYAN, C_INDIGO]
PALETTE_ALL = [C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE, C_TEAL, C_PINK,
               C_AMBER, C_INDIGO, C_CYAN, C_LIME, C_BROWN, C_BLUE]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report_figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {name}")


def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw = pd.read_csv(os.path.join(base, 'data', 'raw', 'spotifyDataset.csv'))
    proc = pd.read_csv(os.path.join(base, 'data', 'processed', 'processed_data.csv'))
    return raw, proc


# ================================================================
# FIGURE 1: Dataset Overview (Grouped Bar + Annotation)
# ================================================================
def fig01_dataset_overview(raw, proc):
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ['Total\nRecords', 'Unique\nTracks', 'Features', 'Genres']
    before = [len(raw), len(raw) - raw.duplicated('track_id').sum(),
              len(raw.columns), raw['track_genre'].nunique()]
    after = [len(proc), len(proc) - proc.duplicated('track_id').sum(),
             len(proc.columns), proc['track_genre'].nunique()]
    x = np.arange(len(categories))
    w = 0.32
    b1 = ax.bar(x - w/2, before, w, label='Before Cleaning', color=C_RED, alpha=0.85, edgecolor='white', linewidth=1.2)
    b2 = ax.bar(x + w/2, after, w, label='After Cleaning', color=C_BLUE, alpha=0.85, edgecolor='white', linewidth=1.2)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + max(before)*0.01,
                    f'{int(h):,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Count')
    ax.set_title('Figure 1: Dataset Overview Before and After Cleaning')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.4)
    save(fig, 'fig01_dataset_overview.png')


# ================================================================
# FIGURE 2: Data Cleaning Pipeline (Waterfall Chart)
# ================================================================
def fig02_cleaning_pipeline(raw, proc):
    fig, ax = plt.subplots(figsize=(9, 5))
    raw_n = len(raw)
    # Compute step-by-step
    dup_id = raw.duplicated('track_id').sum()
    after_id = raw_n - dup_id
    raw['nn'] = raw['track_name'].str.lower().str.strip()
    raw['an'] = raw['artists'].str.lower().str.strip()
    temp = raw.drop_duplicates(subset=['track_id'])
    dup_sem = temp.duplicated(subset=['nn','an']).sum()
    after_sem = after_id - dup_sem
    missing_drop = after_sem - len(proc)
    final = len(proc)

    steps = ['Raw\nDataset', 'ID Duplicates\nRemoved', 'Semantic Dups\nRemoved', 'Missing Values\nDropped', 'Final\nDataset']
    changes = [raw_n, -dup_id, -dup_sem, -missing_drop, 0]
    running = [raw_n, after_id, after_sem, final, final]
    colors = [C_BLUE, C_RED, C_ORANGE, C_AMBER, C_GREEN]
    bottoms = [0, running[1], running[2], running[3], 0]
    heights = [raw_n, dup_id, dup_sem, missing_drop, final]

    bars = ax.bar(steps, heights, bottom=bottoms, color=colors, edgecolor='white', linewidth=1.5, width=0.55)
    # connectors
    for i in range(len(steps)-2):
        ax.plot([i+0.28, i+0.72], [running[i+1], running[i+1]], color='#999', linewidth=1, linestyle='--')
    # labels
    labels_text = [f'{raw_n:,}', f'-{dup_id:,}', f'-{dup_sem:,}', f'-{missing_drop:,}', f'{final:,}']
    for bar, txt, bot, h in zip(bars, labels_text, bottoms, heights):
        y = bot + h/2
        ax.text(bar.get_x() + bar.get_width()/2, y, txt, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

    ax.set_ylabel('Number of Records')
    ax.set_title('Figure 2: Data Cleaning Pipeline (Waterfall)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    # clean up temp cols
    raw.drop(columns=['nn','an'], inplace=True, errors='ignore')
    save(fig, 'fig02_cleaning_pipeline.png')


# ================================================================
# FIGURE 3: Feature Types (Donut Chart)
# ================================================================
def fig03_feature_types(raw):
    fig, ax = plt.subplots(figsize=(6, 6))
    dtypes_map = {'float64': 'Float (9)', 'int64': 'Integer (6)', 'object': 'String (5)', 'bool': 'Boolean (1)'}
    counts = raw.dtypes.value_counts()
    labels = [dtypes_map.get(str(k), str(k)) for k in counts.index]
    colors = [C_BLUE, C_GREEN, C_ORANGE, C_PURPLE][:len(counts)]
    wedges, texts, autotexts = ax.pie(counts.values, labels=labels, colors=colors,
                                       autopct='%1.0f%%', startangle=90, pctdistance=0.78,
                                       wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2.5))
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight('bold')
    ax.set_title('Figure 3: Distribution of Feature Data Types')
    save(fig, 'fig03_feature_types.png')


# ================================================================
# FIGURE 4: Missing Values (Lollipop Chart)
# ================================================================
def fig04_missing_values(raw):
    fig, ax = plt.subplots(figsize=(8, 5))
    missing = raw.isnull().sum().sort_values(ascending=True)
    top = missing.tail(10)
    y = np.arange(len(top))
    colors = [C_RED if v > 0 else C_GREEN for v in top.values]
    ax.hlines(y, 0, top.values, color='#CCCCCC', linewidth=2, zorder=1)
    ax.scatter(top.values, y, color=colors, s=80, zorder=3, edgecolors='white', linewidths=1.5)
    for yi, val in zip(y, top.values):
        ax.text(val + 0.3, yi, str(val), va='center', fontsize=9, fontweight='bold',
                color=C_RED if val > 0 else C_GREEN)
    ax.set_yticks(y)
    ax.set_yticklabels(top.index)
    ax.set_xlabel('Number of Missing Values')
    ax.set_title('Figure 4: Missing Values per Feature (Raw Dataset)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    save(fig, 'fig04_missing_values.png')


# ================================================================
# FIGURE 5: Genre Distribution (Horizontal Bar)
# ================================================================
def fig05_genre_distribution(raw):
    fig, ax = plt.subplots(figsize=(9, 6))
    top15 = raw['track_genre'].value_counts().head(15)
    colors = sns.color_palette('Spectral', len(top15))
    bars = ax.barh(top15.index[::-1], top15.values[::-1], color=colors[::-1],
                   edgecolor='white', linewidth=0.8, height=0.65)
    for bar, val in zip(bars, top15.values[::-1]):
        ax.text(val + top15.values.max()*0.01, bar.get_y() + bar.get_height()/2,
                f'{val:,}', va='center', fontsize=8, fontweight='bold', color='#333')
    ax.set_xlabel('Number of Tracks')
    ax.set_title('Figure 5: Top 15 Genre Distribution')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    save(fig, 'fig05_genre_distribution.png')


# ================================================================
# FIGURE 6: Feature Distributions BEFORE Cleaning (Histograms + KDE)
# ================================================================
def fig06_distributions_before(raw):
    features = ['popularity', 'danceability', 'energy', 'loudness',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo']
    avail = [f for f in features if f in raw.columns]
    n = len(avail)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows*3.5))
    axes = axes.flatten()
    colors = PALETTE_ALL[:n]
    for i, col in enumerate(avail):
        ax = axes[i]
        data = raw[col].dropna()
        sns.histplot(data, ax=ax, kde=True, color=colors[i], alpha=0.6,
                     edgecolor='white', linewidth=0.4, bins=40, stat='density')
        m = data.mean()
        ax.axvline(m, color=C_RED, linestyle='--', linewidth=1.5, label=f'Mean: {m:.2f}')
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.legend(fontsize=7, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Figure 6: Feature Distributions Before Cleaning (Raw Data)', fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, 'fig06_distributions_before.png')


# ================================================================
# FIGURE 7: Feature Distributions AFTER Cleaning (Histograms + KDE)
# ================================================================
def fig07_distributions_after(proc):
    features = ['popularity', 'danceability', 'energy', 'loudness',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo']
    avail = [f for f in features if f in proc.columns]
    n = len(avail)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows*3.5))
    axes = axes.flatten()
    colors = PALETTE_ALL[:n]
    for i, col in enumerate(avail):
        ax = axes[i]
        data = proc[col].dropna()
        sns.histplot(data, ax=ax, kde=True, color=colors[i], alpha=0.6,
                     edgecolor='white', linewidth=0.4, bins=40, stat='density')
        m = data.mean()
        ax.axvline(m, color=C_RED, linestyle='--', linewidth=1.5, label=f'Mean: {m:.2f}')
        ax.set_title(f'{col} (Scaled)', fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.legend(fontsize=7, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Figure 7: Feature Distributions After Cleaning (Scaled Data)', fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, 'fig07_distributions_after.png')


# ================================================================
# FIGURE 8: Correlation Heatmap BEFORE
# ================================================================
def fig08_correlation_before(raw):
    features = ['popularity', 'danceability', 'energy', 'loudness',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo']
    avail = [f for f in features if f in raw.columns]
    fig, ax = plt.subplots(figsize=(9, 8))
    corr = raw[avail].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, ax=ax, square=True, linewidths=0.8, linecolor='white',
                cbar_kws={'shrink': 0.8}, annot_kws={'size': 9})
    ax.set_title('Figure 8: Feature Correlation Matrix (Before Cleaning)', fontsize=14, fontweight='bold', pad=16)
    save(fig, 'fig08_correlation_before.png')


# ================================================================
# FIGURE 9: Correlation Heatmap AFTER
# ================================================================
def fig09_correlation_after(proc):
    features = ['popularity', 'danceability', 'energy', 'loudness',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo']
    avail = [f for f in features if f in proc.columns]
    fig, ax = plt.subplots(figsize=(9, 8))
    corr = proc[avail].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, ax=ax, square=True, linewidths=0.8, linecolor='white',
                cbar_kws={'shrink': 0.8}, annot_kws={'size': 9})
    ax.set_title('Figure 9: Feature Correlation Matrix (After Cleaning)', fontsize=14, fontweight='bold', pad=16)
    save(fig, 'fig09_correlation_after.png')


# ================================================================
# FIGURE 10: Boxplot Comparison Before vs After (Side by Side)
# ================================================================
def fig10_boxplot_comparison(raw, proc):
    features = ['danceability', 'energy', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence']
    avail = [f for f in features if f in raw.columns and f in proc.columns]
    fig, axes = plt.subplots(1, len(avail), figsize=(len(avail)*2.8, 6))
    for i, feat in enumerate(avail):
        ax = axes[i]
        bp = ax.boxplot([raw[feat].dropna(), proc[feat].dropna()],
                        labels=['Before', 'After'], patch_artist=True,
                        widths=0.5,
                        boxprops=dict(linewidth=1.5),
                        whiskerprops=dict(linewidth=1.2),
                        medianprops=dict(color=C_RED, linewidth=2),
                        capprops=dict(linewidth=1.2))
        bp['boxes'][0].set_facecolor(C_RED)
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor(C_BLUE)
        bp['boxes'][1].set_alpha(0.5)
        ax.set_title(feat, fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Figure 10: Feature Boxplots Before vs After Scaling', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig10_boxplot_comparison.png')


# ================================================================
# FIGURE 11: Scaling Effect (KDE Overlay)
# ================================================================
def fig11_scaling_effect(raw, proc):
    features = ['danceability', 'energy', 'loudness', 'tempo']
    avail = [f for f in features if f in raw.columns and f in proc.columns]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for i, feat in enumerate(avail):
        ax = axes[i]
        b = raw[feat].dropna()
        a = proc[feat].dropna()
        b_norm = (b - b.min()) / (b.max() - b.min() + 1e-8)
        a_norm = (a - a.min()) / (a.max() - a.min() + 1e-8)
        sns.kdeplot(b_norm, ax=ax, color=C_RED, fill=True, alpha=0.25, linewidth=2.5, label='Before')
        sns.kdeplot(a_norm, ax=ax, color=C_BLUE, fill=True, alpha=0.25, linewidth=2.5, label='After (Scaled)')
        ax.set_title(feat.upper(), fontsize=12, fontweight='bold')
        ax.set_xlabel('Normalized Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3)
    fig.suptitle('Figure 11: Feature Scaling Effect (Before vs After)', fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, 'fig11_scaling_effect.png')


# ================================================================
# FIGURE 12: Popularity Analysis (KDE + CDF + Violin)
# ================================================================
def fig12_popularity_analysis(raw, proc):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # KDE overlay
    ax = axes[0]
    sns.kdeplot(raw['popularity'].dropna(), ax=ax, color=C_RED, fill=True, alpha=0.25, linewidth=2, label='Before')
    sns.kdeplot(proc['popularity'].dropna(), ax=ax, color=C_BLUE, fill=True, alpha=0.25, linewidth=2, label='After')
    ax.set_title('Popularity KDE', fontweight='bold')
    ax.set_xlabel('Popularity')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # CDF
    ax2 = axes[1]
    for df, color, label in [(raw, C_RED, 'Before'), (proc, C_BLUE, 'After')]:
        data = np.sort(df['popularity'].dropna().values)
        cdf = np.arange(1, len(data)+1) / len(data)
        step = max(1, len(data)//2000)
        ax2.plot(data[::step], cdf[::step], color=color, linewidth=2.5, label=label)
    ax2.set_title('Cumulative Distribution', fontweight='bold')
    ax2.set_xlabel('Popularity')
    ax2.set_ylabel('Cumulative Fraction')
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(alpha=0.3)

    # Violin
    ax3 = axes[2]
    combined = pd.DataFrame({
        'Popularity': pd.concat([
            raw['popularity'].dropna().sample(min(5000, len(raw)), random_state=42),
            proc['popularity'].dropna().sample(min(5000, len(proc)), random_state=42)
        ]),
        'Dataset': ['Before']*min(5000, len(raw)) + ['After']*min(5000, len(proc))
    })
    sns.violinplot(data=combined, x='Dataset', y='Popularity', palette=[C_RED, C_BLUE],
                   ax=ax3, inner='quartile', linewidth=1.5)
    ax3.set_title('Popularity Violin', fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    fig.suptitle('Figure 12: Popularity Distribution Analysis', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig12_popularity_analysis.png')


# ================================================================
# FIGURE 13: Outlier Detection (Box + Percentage badges)
# ================================================================
def fig13_outlier_detection(raw, proc):
    features = ['danceability', 'energy', 'loudness', 'tempo', 'duration_ms']
    avail = [f for f in features if f in raw.columns]
    fig, axes = plt.subplots(2, len(avail), figsize=(len(avail)*3.5, 9))
    for idx, feat in enumerate(avail):
        for row, (df, color, label) in enumerate([(raw, C_RED, 'Before'), (proc, C_BLUE, 'After')]):
            ax = axes[row, idx]
            data = df[feat].dropna()
            Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_pct = ((data < Q1-1.5*IQR) | (data > Q3+1.5*IQR)).mean() * 100
            bp = ax.boxplot(data.values, patch_artist=True, widths=0.5,
                            boxprops=dict(facecolor=color, alpha=0.45, edgecolor=color, linewidth=1.5),
                            whiskerprops=dict(color=color, linewidth=1.3),
                            capprops=dict(color=color, linewidth=1.3),
                            medianprops=dict(color='#333', linewidth=2),
                            flierprops=dict(marker='.', markerfacecolor=C_ORANGE, markersize=3, alpha=0.5))
            ax.text(0.95, 0.95, f'{outlier_pct:.1f}%\noutliers', transform=ax.transAxes,
                    ha='right', va='top', fontsize=8, fontweight='bold',
                    color=C_RED if outlier_pct > 2 else C_GREEN,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#CCC'))
            ax.set_title(f'{feat}\n({label})', fontsize=9, fontweight='bold')
            ax.set_xticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Figure 13: Outlier Detection Before vs After Cleaning', fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, 'fig13_outlier_detection.png')


# ================================================================
# FIGURE 14: Pairplot (Key Features, Scatter Matrix)
# ================================================================
def fig14_pairplot(proc):
    features = ['danceability', 'energy', 'valence', 'acousticness', 'loudness']
    avail = [f for f in features if f in proc.columns]
    sample = proc[avail].dropna().sample(min(2000, len(proc)), random_state=42)
    g = sns.pairplot(sample, diag_kind='kde', corner=True,
                     plot_kws={'alpha': 0.35, 'color': C_BLUE, 's': 12, 'edgecolor': 'none'},
                     diag_kws={'color': C_BLUE, 'fill': True, 'alpha': 0.4, 'linewidth': 2})
    g.fig.suptitle('Figure 14: Pairwise Feature Relationships (Processed Data)', fontsize=15, fontweight='bold', y=1.02)
    save(g.fig, 'fig14_pairplot.png')


# ================================================================
# MAIN
# ================================================================
def main():
    print("\n" + "="*60)
    print("  GENERATING REPORT FIGURES")
    print("="*60 + "\n")

    raw, proc = load_data()
    print(f"  Raw: {len(raw):,} records | Processed: {len(proc):,} records\n")

    fig01_dataset_overview(raw, proc)
    fig02_cleaning_pipeline(raw, proc)
    fig03_feature_types(raw)
    fig04_missing_values(raw)
    fig05_genre_distribution(raw)
    fig06_distributions_before(raw)
    fig07_distributions_after(proc)
    fig08_correlation_before(raw)
    fig09_correlation_after(proc)
    fig10_boxplot_comparison(raw, proc)
    fig11_scaling_effect(raw, proc)
    fig12_popularity_analysis(raw, proc)
    fig13_outlier_detection(raw, proc)
    fig14_pairplot(proc)

    print(f"\n  All 14 figures saved to: {OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
