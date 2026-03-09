"""
Spotify Dataset EDA Visualizations
Creates colorful visualizations comparing data before and after cleaning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for colorful visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Custom color palettes
BEFORE_PALETTE = sns.color_palette("rocket", 10)
AFTER_PALETTE = sns.color_palette("mako", 10)
COMPARISON_PALETTE = ["#FF6B6B", "#4ECDC4"]  # Red for before, Teal for after

# Create output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load raw and processed datasets"""
    raw_path = os.path.join(os.path.dirname(OUTPUT_DIR), 'data', 'raw', 'spotifyDataset.csv')
    processed_path = os.path.join(os.path.dirname(OUTPUT_DIR), 'data', 'processed', 'processed_data.csv')
    
    df_raw = pd.read_csv(raw_path)
    df_processed = pd.read_csv(processed_path)
    
    return df_raw, df_processed


def plot_data_size_comparison(df_raw, df_processed):
    """Bar chart comparing dataset sizes before and after cleaning"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Dataset size comparison
    sizes = [len(df_raw), len(df_processed)]
    labels = ['Before Cleaning', 'After Cleaning']
    colors = COMPARISON_PALETTE
    
    bars = axes[0].bar(labels, sizes, color=colors, edgecolor='white', linewidth=2)
    axes[0].set_title('Dataset Size Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Records', fontsize=12)
    
    # Add value labels on bars
    for bar, size in zip(bars, sizes):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                    f'{size:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Records removed pie chart
    removed = len(df_raw) - len(df_processed)
    kept = len(df_processed)
    pie_colors = ['#FF6B6B', '#4ECDC4']
    
    wedges, texts, autotexts = axes[1].pie(
        [removed, kept], 
        labels=['Removed', 'Kept'],
        colors=pie_colors,
        autopct='%1.1f%%',
        explode=(0.05, 0),
        shadow=True,
        startangle=90
    )
    autotexts[0].set_fontweight('bold')
    autotexts[1].set_fontweight('bold')
    axes[1].set_title(f'Data Cleaning Impact\n({removed:,} records removed)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_data_size_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 01_data_size_comparison.png")


def plot_missing_values_heatmap(df_raw, df_processed):
    """Heatmap showing missing values before and after"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Before cleaning
    missing_before = df_raw.isnull().sum()
    missing_before_pct = (missing_before / len(df_raw) * 100).sort_values(ascending=False)
    
    # Create heatmap data for before
    sns.barplot(x=missing_before_pct.values[:15], y=missing_before_pct.index[:15], 
                palette='Reds_r', ax=axes[0])
    axes[0].set_title('Missing Values - BEFORE Cleaning', fontsize=14, fontweight='bold', color='#FF6B6B')
    axes[0].set_xlabel('Percentage Missing (%)', fontsize=12)
    axes[0].set_ylabel('Features', fontsize=12)
    
    # After cleaning
    missing_after = df_processed.isnull().sum()
    missing_after_pct = (missing_after / len(df_processed) * 100).sort_values(ascending=False)
    
    sns.barplot(x=missing_after_pct.values[:15], y=missing_after_pct.index[:15], 
                palette='Greens_r', ax=axes[1])
    axes[1].set_title('Missing Values - AFTER Cleaning', fontsize=14, fontweight='bold', color='#4ECDC4')
    axes[1].set_xlabel('Percentage Missing (%)', fontsize=12)
    axes[1].set_ylabel('Features', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_missing_values_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 02_missing_values_comparison.png")


def plot_numeric_distributions_before(df_raw):
    """Distribution plots for numeric features BEFORE cleaning"""
    numeric_cols = ['popularity', 'danceability', 'energy', 'loudness', 
                   'speechiness', 'acousticness', 'instrumentalness', 
                   'liveness', 'valence', 'tempo', 'duration_ms']
    
    available_cols = [col for col in numeric_cols if col in df_raw.columns]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    colors = sns.color_palette("rocket", len(available_cols))
    
    for idx, col in enumerate(available_cols):
        data = df_raw[col].dropna()
        
        # Histogram with KDE
        sns.histplot(data, ax=axes[idx], kde=True, color=colors[idx], 
                    alpha=0.7, edgecolor='white', linewidth=0.5)
        axes[idx].set_title(f'{col.upper()}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('')
        
        # Add statistics
        mean_val = data.mean()
        axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        axes[idx].legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(available_cols), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Feature Distributions - BEFORE Cleaning (Raw Data)', 
                fontsize=16, fontweight='bold', color='#FF6B6B', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_distributions_before_cleaning.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 03_distributions_before_cleaning.png")


def plot_numeric_distributions_after(df_processed):
    """Distribution plots for numeric features AFTER cleaning"""
    numeric_cols = ['popularity', 'danceability', 'energy', 'loudness', 
                   'speechiness', 'acousticness', 'instrumentalness', 
                   'liveness', 'valence', 'tempo', 'duration_ms']
    
    available_cols = [col for col in numeric_cols if col in df_processed.columns]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    colors = sns.color_palette("mako", len(available_cols))
    
    for idx, col in enumerate(available_cols):
        data = df_processed[col].dropna()
        
        # Histogram with KDE
        sns.histplot(data, ax=axes[idx], kde=True, color=colors[idx], 
                    alpha=0.7, edgecolor='white', linewidth=0.5)
        axes[idx].set_title(f'{col.upper()} (Scaled)', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('')
        
        # Add statistics
        mean_val = data.mean()
        axes[idx].axvline(mean_val, color='navy', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        axes[idx].legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(available_cols), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Feature Distributions - AFTER Cleaning (Processed & Scaled Data)', 
                fontsize=16, fontweight='bold', color='#4ECDC4', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_distributions_after_cleaning.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 04_distributions_after_cleaning.png")


def plot_boxplots_comparison(df_raw, df_processed):
    """Side-by-side boxplots comparing distributions"""
    features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                'instrumentalness', 'liveness', 'valence']
    
    available_features = [f for f in features if f in df_raw.columns and f in df_processed.columns]
    
    fig, axes = plt.subplots(2, len(available_features), figsize=(20, 10))
    
    for idx, feature in enumerate(available_features):
        # Before - top row
        box1 = axes[0, idx].boxplot(df_raw[feature].dropna(), patch_artist=True)
        box1['boxes'][0].set_facecolor('#FF6B6B')
        box1['boxes'][0].set_alpha(0.7)
        axes[0, idx].set_title(f'{feature}\n(Before)', fontsize=10, fontweight='bold')
        axes[0, idx].set_ylabel('Value')
        
        # After - bottom row
        box2 = axes[1, idx].boxplot(df_processed[feature].dropna(), patch_artist=True)
        box2['boxes'][0].set_facecolor('#4ECDC4')
        box2['boxes'][0].set_alpha(0.7)
        axes[1, idx].set_title(f'{feature}\n(After)', fontsize=10, fontweight='bold')
        axes[1, idx].set_ylabel('Scaled Value')
    
    fig.suptitle('Feature Boxplots: Before vs After Cleaning', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_boxplots_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 05_boxplots_comparison.png")


def plot_correlation_heatmaps(df_raw, df_processed):
    """Correlation heatmaps before and after cleaning"""
    numeric_cols = ['popularity', 'danceability', 'energy', 'loudness', 
                   'speechiness', 'acousticness', 'instrumentalness', 
                   'liveness', 'valence', 'tempo']
    
    available_cols_raw = [col for col in numeric_cols if col in df_raw.columns]
    available_cols_proc = [col for col in numeric_cols if col in df_processed.columns]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Before cleaning correlation
    corr_before = df_raw[available_cols_raw].corr()
    mask = np.triu(np.ones_like(corr_before, dtype=bool))
    sns.heatmap(corr_before, mask=mask, annot=True, fmt='.2f', 
                cmap='RdYlBu_r', center=0, ax=axes[0],
                square=True, linewidths=0.5)
    axes[0].set_title('Feature Correlations - BEFORE Cleaning', 
                     fontsize=14, fontweight='bold', color='#FF6B6B')
    
    # After cleaning correlation
    corr_after = df_processed[available_cols_proc].corr()
    mask = np.triu(np.ones_like(corr_after, dtype=bool))
    sns.heatmap(corr_after, mask=mask, annot=True, fmt='.2f', 
                cmap='RdYlBu_r', center=0, ax=axes[1],
                square=True, linewidths=0.5)
    axes[1].set_title('Feature Correlations - AFTER Cleaning', 
                     fontsize=14, fontweight='bold', color='#4ECDC4')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '06_correlation_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 06_correlation_heatmaps.png")


def plot_genre_distribution(df_raw, df_processed):
    """Genre distribution comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Before cleaning - Top 15 genres
    genre_counts_before = df_raw['track_genre'].value_counts().head(15)
    colors_before = sns.color_palette("rocket", len(genre_counts_before))
    
    bars1 = axes[0].barh(genre_counts_before.index[::-1], genre_counts_before.values[::-1], 
                        color=colors_before[::-1], edgecolor='white', linewidth=0.5)
    axes[0].set_title('Top 15 Genres - BEFORE Cleaning', fontsize=14, fontweight='bold', color='#FF6B6B')
    axes[0].set_xlabel('Number of Tracks', fontsize=12)
    
    # Add value labels
    for bar, count in zip(bars1, genre_counts_before.values[::-1]):
        axes[0].text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, 
                    f'{count:,}', va='center', fontsize=9)
    
    # After cleaning - Top 15 genres
    genre_counts_after = df_processed['track_genre'].value_counts().head(15)
    colors_after = sns.color_palette("mako", len(genre_counts_after))
    
    bars2 = axes[1].barh(genre_counts_after.index[::-1], genre_counts_after.values[::-1], 
                        color=colors_after[::-1], edgecolor='white', linewidth=0.5)
    axes[1].set_title('Top 15 Genres - AFTER Cleaning', fontsize=14, fontweight='bold', color='#4ECDC4')
    axes[1].set_xlabel('Number of Tracks', fontsize=12)
    
    # Add value labels
    for bar, count in zip(bars2, genre_counts_after.values[::-1]):
        axes[1].text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, 
                    f'{count:,}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '07_genre_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 07_genre_distribution.png")


def plot_popularity_analysis(df_raw, df_processed):
    """Popularity distribution analysis"""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig)
    
    # Histogram comparison
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(df_raw['popularity'].dropna(), ax=ax1, kde=True, color='#FF6B6B', 
                alpha=0.6, label='Before', edgecolor='white')
    ax1.set_title('Popularity Distribution\n(Before Cleaning)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Popularity')
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(df_processed['popularity'].dropna(), ax=ax2, kde=True, color='#4ECDC4', 
                alpha=0.6, label='After', edgecolor='white')
    ax2.set_title('Popularity Distribution\n(After Cleaning - Scaled)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Popularity (Scaled)')
    ax2.legend()
    
    # Violin plot comparison
    ax3 = fig.add_subplot(gs[0, 2])
    data_violin = pd.DataFrame({
        'Popularity': list(df_raw['popularity'].dropna()[:10000]) + list(df_processed['popularity'].dropna()[:10000]),
        'Dataset': ['Before']*len(df_raw['popularity'].dropna()[:10000]) + ['After']*len(df_processed['popularity'].dropna()[:10000])
    })
    sns.violinplot(data=data_violin, x='Dataset', y='Popularity', palette=COMPARISON_PALETTE, ax=ax3)
    ax3.set_title('Popularity Violin Plot\nComparison', fontsize=12, fontweight='bold')
    
    # Popularity by genre (before)
    ax4 = fig.add_subplot(gs[1, :])
    top_genres = df_raw['track_genre'].value_counts().head(10).index
    genre_pop_before = df_raw[df_raw['track_genre'].isin(top_genres)].groupby('track_genre')['popularity'].mean()
    genre_pop_before = genre_pop_before.reindex(top_genres)
    
    x = np.arange(len(top_genres))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, genre_pop_before.values, width, label='Before Cleaning', 
                   color='#FF6B6B', alpha=0.8, edgecolor='white')
    
    # For after, we need to use raw popularity values since processed is scaled
    # Just show the before data for genre comparison
    ax4.set_xlabel('Genre', fontsize=12)
    ax4.set_ylabel('Average Popularity', fontsize=12)
    ax4.set_title('Average Popularity by Top 10 Genres (Raw Data)', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(top_genres, rotation=45, ha='right')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '08_popularity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 08_popularity_analysis.png")


def plot_outlier_detection(df_raw, df_processed):
    """Visualize outliers before and after cleaning"""
    features = ['danceability', 'energy', 'loudness', 'tempo', 'duration_ms']
    available_features = [f for f in features if f in df_raw.columns]
    
    fig, axes = plt.subplots(2, len(available_features), figsize=(20, 10))
    
    for idx, feature in enumerate(available_features):
        # Before - strip plot with outliers highlighted
        data_before = df_raw[feature].dropna()
        Q1 = data_before.quantile(0.25)
        Q3 = data_before.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (data_before < Q1 - 1.5*IQR) | (data_before > Q3 + 1.5*IQR)
        outlier_count_before = outlier_mask.sum()
        
        axes[0, idx].scatter(range(len(data_before[~outlier_mask][:1000])), 
                            data_before[~outlier_mask][:1000], 
                            alpha=0.3, c='#FF6B6B', s=10, label='Normal')
        axes[0, idx].scatter(range(len(data_before[outlier_mask][:500])), 
                            data_before[outlier_mask][:500], 
                            alpha=0.7, c='yellow', s=15, label=f'Outliers ({outlier_count_before:,})')
        axes[0, idx].set_title(f'{feature}\n(Before)', fontsize=10, fontweight='bold')
        axes[0, idx].legend(fontsize=8)
        axes[0, idx].set_ylabel('Value')
        
        # After
        data_after = df_processed[feature].dropna()
        Q1 = data_after.quantile(0.25)
        Q3 = data_after.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask_after = (data_after < Q1 - 1.5*IQR) | (data_after > Q3 + 1.5*IQR)
        outlier_count_after = outlier_mask_after.sum()
        
        axes[1, idx].scatter(range(len(data_after[~outlier_mask_after][:1000])), 
                            data_after[~outlier_mask_after][:1000], 
                            alpha=0.3, c='#4ECDC4', s=10, label='Normal')
        axes[1, idx].scatter(range(len(data_after[outlier_mask_after][:500])), 
                            data_after[outlier_mask_after][:500], 
                            alpha=0.7, c='orange', s=15, label=f'Outliers ({outlier_count_after:,})')
        axes[1, idx].set_title(f'{feature}\n(After)', fontsize=10, fontweight='bold')
        axes[1, idx].legend(fontsize=8)
        axes[1, idx].set_ylabel('Scaled Value')
    
    fig.suptitle('Outlier Detection: Before vs After Cleaning', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '09_outlier_detection.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 09_outlier_detection.png")


def plot_pairplot_before(df_raw):
    """Pairplot for key features before cleaning"""
    features = ['danceability', 'energy', 'valence', 'loudness', 'tempo']
    available_features = [f for f in features if f in df_raw.columns]
    
    # Sample for performance
    sample = df_raw[available_features].dropna().sample(min(2000, len(df_raw)))
    
    g = sns.pairplot(sample, diag_kind='kde', plot_kws={'alpha': 0.5, 'color': '#FF6B6B'},
                    diag_kws={'color': '#FF6B6B', 'fill': True, 'alpha': 0.5})
    g.fig.suptitle('Feature Pairplot - BEFORE Cleaning', fontsize=16, fontweight='bold', 
                  color='#FF6B6B', y=1.02)
    
    plt.savefig(os.path.join(OUTPUT_DIR, '10_pairplot_before.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 10_pairplot_before.png")


def plot_pairplot_after(df_processed):
    """Pairplot for key features after cleaning"""
    features = ['danceability', 'energy', 'valence', 'loudness', 'tempo']
    available_features = [f for f in features if f in df_processed.columns]
    
    # Sample for performance
    sample = df_processed[available_features].dropna().sample(min(2000, len(df_processed)))
    
    g = sns.pairplot(sample, diag_kind='kde', plot_kws={'alpha': 0.5, 'color': '#4ECDC4'},
                    diag_kws={'color': '#4ECDC4', 'fill': True, 'alpha': 0.5})
    g.fig.suptitle('Feature Pairplot - AFTER Cleaning', fontsize=16, fontweight='bold', 
                  color='#4ECDC4', y=1.02)
    
    plt.savefig(os.path.join(OUTPUT_DIR, '11_pairplot_after.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 11_pairplot_after.png")


def plot_duplicates_analysis(df_raw, df_processed):
    """Visualize duplicate records analysis"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Duplicate track_ids
    dup_ids_before = df_raw.duplicated(subset=['track_id']).sum()
    unique_ids_before = len(df_raw) - dup_ids_before
    
    dup_ids_after = df_processed.duplicated(subset=['track_id']).sum()
    unique_ids_after = len(df_processed) - dup_ids_after
    
    # Pie chart for before
    colors = ['#FF6B6B', '#FFE66D']
    axes[0].pie([unique_ids_before, dup_ids_before], labels=['Unique', 'Duplicates'],
               colors=colors, autopct='%1.1f%%', explode=(0, 0.1), shadow=True)
    axes[0].set_title(f'Track ID Duplicates\nBEFORE ({dup_ids_before:,} dups)', 
                     fontsize=12, fontweight='bold', color='#FF6B6B')
    
    # Pie chart for after
    colors = ['#4ECDC4', '#95E1D3']
    axes[1].pie([unique_ids_after, dup_ids_after], labels=['Unique', 'Duplicates'],
               colors=colors, autopct='%1.1f%%', explode=(0, 0.1), shadow=True)
    axes[1].set_title(f'Track ID Duplicates\nAFTER ({dup_ids_after:,} dups)', 
                     fontsize=12, fontweight='bold', color='#4ECDC4')
    
    # Bar comparison
    x = ['Duplicates Removed', 'Records Cleaned']
    values = [dup_ids_before - dup_ids_after, len(df_raw) - len(df_processed)]
    colors = ['#FFE66D', '#95E1D3']
    
    bars = axes[2].bar(x, values, color=colors, edgecolor='white', linewidth=2)
    axes[2].set_title('Cleaning Impact Summary', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Number of Records')
    
    for bar, val in zip(bars, values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
                    f'{val:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '12_duplicates_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 12_duplicates_analysis.png")


def plot_feature_scaling_effect(df_raw, df_processed):
    """Show the effect of feature scaling"""
    features = ['danceability', 'energy', 'loudness', 'tempo']
    available_features = [f for f in features if f in df_raw.columns and f in df_processed.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(available_features):
        # Create overlaid KDE plots
        ax = axes[idx]
        
        data_before = df_raw[feature].dropna()
        data_after = df_processed[feature].dropna()
        
        # Normalize before data to [0, 1] for comparison
        data_before_norm = (data_before - data_before.min()) / (data_before.max() - data_before.min())
        
        sns.kdeplot(data_before_norm, ax=ax, color='#FF6B6B', fill=True, alpha=0.4, 
                   label='Before (normalized)', linewidth=2)
        
        # Normalize after data to [0, 1] for visualization
        data_after_norm = (data_after - data_after.min()) / (data_after.max() - data_after.min())
        sns.kdeplot(data_after_norm, ax=ax, color='#4ECDC4', fill=True, alpha=0.4, 
                   label='After (scaled)', linewidth=2)
        
        ax.set_title(f'{feature.upper()}\nScaling Effect', fontsize=12, fontweight='bold')
        ax.set_xlabel('Normalized Value')
        ax.set_ylabel('Density')
        ax.legend()
    
    fig.suptitle('Feature Scaling Effect: Before vs After', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '13_scaling_effect.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 13_scaling_effect.png")


def plot_summary_dashboard(df_raw, df_processed):
    """Create a summary dashboard"""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Dataset info box
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    info_text = f"""
    📊 DATASET SUMMARY
    
    BEFORE CLEANING:
    • Records: {len(df_raw):,}
    • Features: {len(df_raw.columns)}
    • Genres: {df_raw['track_genre'].nunique()}
    
    AFTER CLEANING:
    • Records: {len(df_processed):,}
    • Features: {len(df_processed.columns)}
    • Reduction: {(1 - len(df_processed)/len(df_raw))*100:.1f}%
    """
    ax1.text(0.1, 0.5, info_text, transform=ax1.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax1.set_title('Overview', fontsize=14, fontweight='bold')
    
    # Data types
    ax2 = fig.add_subplot(gs[0, 1])
    dtypes = df_raw.dtypes.value_counts()
    colors = sns.color_palette("husl", len(dtypes))
    ax2.pie(dtypes.values, labels=dtypes.index.astype(str), colors=colors, autopct='%1.0f%%')
    ax2.set_title('Data Types Distribution', fontsize=12, fontweight='bold')
    
    # Top 5 genres
    ax3 = fig.add_subplot(gs[0, 2:])
    top5 = df_raw['track_genre'].value_counts().head(5)
    colors = sns.color_palette("rocket", 5)
    ax3.barh(top5.index[::-1], top5.values[::-1], color=colors[::-1])
    ax3.set_title('Top 5 Genres', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Count')
    
    # Key metrics comparison
    ax4 = fig.add_subplot(gs[1, :2])
    metrics = ['Mean Popularity', 'Mean Energy', 'Mean Danceability']
    before_vals = [df_raw['popularity'].mean(), df_raw['energy'].mean(), df_raw['danceability'].mean()]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax4.bar(x, before_vals, width, label='Before', color='#FF6B6B', alpha=0.8)
    ax4.set_ylabel('Value')
    ax4.set_title('Key Metrics (Raw Values)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    
    # Explicit content
    ax5 = fig.add_subplot(gs[1, 2])
    if 'explicit' in df_raw.columns:
        explicit_counts = df_raw['explicit'].value_counts()
        colors = ['#4ECDC4', '#FF6B6B']
        ax5.pie(explicit_counts.values, labels=['Clean', 'Explicit'], colors=colors, 
               autopct='%1.1f%%', explode=(0, 0.05))
    ax5.set_title('Explicit Content (Before)', fontsize=12, fontweight='bold')
    
    # Time signature distribution
    ax6 = fig.add_subplot(gs[1, 3])
    if 'time_signature' in df_raw.columns:
        ts_counts = df_raw['time_signature'].value_counts()
        colors = sns.color_palette("mako", len(ts_counts))
        ax6.pie(ts_counts.values, labels=ts_counts.index, colors=colors, autopct='%1.0f%%')
    ax6.set_title('Time Signatures', fontsize=12, fontweight='bold')
    
    # Feature ranges
    ax7 = fig.add_subplot(gs[2, :])
    features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness']
    available_features = [f for f in features if f in df_raw.columns]
    
    stats_before = df_raw[available_features].describe().loc[['min', 'max', 'mean']]
    
    x = np.arange(len(available_features))
    width = 0.25
    
    ax7.bar(x - width, stats_before.loc['min'], width, label='Min', color='#FF6B6B', alpha=0.7)
    ax7.bar(x, stats_before.loc['mean'], width, label='Mean', color='#FFE66D', alpha=0.7)
    ax7.bar(x + width, stats_before.loc['max'], width, label='Max', color='#4ECDC4', alpha=0.7)
    
    ax7.set_ylabel('Value')
    ax7.set_title('Feature Ranges (Before Cleaning)', fontsize=12, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(available_features, rotation=45, ha='right')
    ax7.legend()
    
    fig.suptitle('📈 Spotify Dataset EDA Summary Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(OUTPUT_DIR, '14_summary_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 14_summary_dashboard.png")


def main():
    """Run all EDA visualizations"""
    print("\n" + "="*60)
    print("🎵 SPOTIFY DATASET EDA VISUALIZATIONS")
    print("="*60 + "\n")
    
    print("Loading datasets...")
    df_raw, df_processed = load_data()
    print(f"✓ Raw data: {len(df_raw):,} records")
    print(f"✓ Processed data: {len(df_processed):,} records\n")
    
    print("Generating visualizations...\n")
    
    # Generate all plots
    plot_data_size_comparison(df_raw, df_processed)
    plot_missing_values_heatmap(df_raw, df_processed)
    plot_numeric_distributions_before(df_raw)
    plot_numeric_distributions_after(df_processed)
    plot_boxplots_comparison(df_raw, df_processed)
    plot_correlation_heatmaps(df_raw, df_processed)
    plot_genre_distribution(df_raw, df_processed)
    plot_popularity_analysis(df_raw, df_processed)
    plot_outlier_detection(df_raw, df_processed)
    plot_pairplot_before(df_raw)
    plot_pairplot_after(df_processed)
    plot_duplicates_analysis(df_raw, df_processed)
    plot_feature_scaling_effect(df_raw, df_processed)
    plot_summary_dashboard(df_raw, df_processed)
    
    print("\n" + "="*60)
    print(f"✅ All visualizations saved to: {OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
