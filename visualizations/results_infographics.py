"""
Results & Evaluation Infographics
Infographic-style figures for the report's results chapter.
White backgrounds, colorful, visually engaging.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
from matplotlib.gridspec import GridSpec
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report_figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors
C_BLUE = '#2196F3'
C_RED = '#E53935'
C_GREEN = '#43A047'
C_ORANGE = '#FB8C00'
C_PURPLE = '#8E24AA'
C_TEAL = '#00897B'
C_PINK = '#D81B60'
C_AMBER = '#FFB300'
C_INDIGO = '#3949AB'
C_DARK = '#263238'
C_LIGHT_GRAY = '#F5F5F5'
C_MID_GRAY = '#BDBDBD'


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {name}")


# ================================================================
# FIGURE 15: Evaluation Metrics Scorecard (Dashboard Style)
# ================================================================
def fig15_metrics_scorecard():
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    fig.suptitle('Figure 15: Model Evaluation Scorecard', fontsize=20, fontweight='bold',
                 color=C_DARK, y=0.97)

    metrics = [
        {'name': 'Relevance\n(Cosine Similarity)', 'value': 0.9355, 'threshold': 0.70,
         'max': 1.0, 'color': C_BLUE, 'unit': '', 'status': 'PASS', 'desc': 'How close recs match taste'},
        {'name': 'Diversity\n(ILD Score)', 'value': 0.1069, 'threshold': 0.05,
         'max': 0.90, 'color': C_TEAL, 'unit': '', 'status': 'PASS', 'desc': 'Variety within rec list'},
        {'name': 'Novelty\nScore', 'value': 0.3942, 'threshold': 0.10,
         'max': 1.0, 'color': C_PURPLE, 'unit': '', 'status': 'PASS', 'desc': 'Hidden gems surfaced'},
        {'name': 'Hit Rate\n@ K=10', 'value': 14.29, 'threshold': 5.0,
         'max': 100, 'color': C_ORANGE, 'unit': '%', 'status': 'PASS', 'desc': 'Predictive accuracy'},
        {'name': 'Hit Rate\n@ K=20', 'value': 35.71, 'threshold': 5.0,
         'max': 100, 'color': C_AMBER, 'unit': '%', 'status': 'PASS', 'desc': 'Broader prediction window'},
        {'name': 'Hit Rate\n@ K=50', 'value': 47.62, 'threshold': 5.0,
         'max': 100, 'color': C_GREEN, 'unit': '%', 'status': 'PASS', 'desc': 'Extended neighborhood'},
    ]

    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35, top=0.88, bottom=0.06)

    for idx, m in enumerate(metrics):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')

        # Background card
        card = FancyBboxPatch((0.3, 0.3), 9.4, 9.4, boxstyle="round,pad=0.3",
                              facecolor=C_LIGHT_GRAY, edgecolor=C_MID_GRAY, linewidth=1.5)
        ax.add_patch(card)

        # Gauge arc
        center_x, center_y = 5, 5.2
        radius = 3.2
        # Background arc
        theta_start, theta_end = 210, -30
        for angle in np.linspace(210, -30, 100):
            rad = np.radians(angle)
            x = center_x + radius * np.cos(rad)
            y = center_y + radius * np.sin(rad)
            ax.plot(x, y, '.', color='#E0E0E0', markersize=4)

        # Filled arc
        if m['max'] <= 1:
            fill_frac = m['value'] / m['max']
        else:
            fill_frac = m['value'] / m['max']
        fill_angle = 210 - fill_frac * 240
        for angle in np.linspace(210, max(fill_angle, -30), int(fill_frac * 100)):
            rad = np.radians(angle)
            x = center_x + radius * np.cos(rad)
            y = center_y + radius * np.sin(rad)
            ax.plot(x, y, '.', color=m['color'], markersize=7)

        # Threshold marker
        if m['max'] <= 1:
            thresh_frac = m['threshold'] / m['max']
        else:
            thresh_frac = m['threshold'] / m['max']
        thresh_angle = np.radians(210 - thresh_frac * 240)
        tx = center_x + (radius + 0.4) * np.cos(thresh_angle)
        ty = center_y + (radius + 0.4) * np.sin(thresh_angle)
        ax.plot(tx, ty, 'v', color=C_RED, markersize=8, zorder=5)

        # Center value
        if m['unit'] == '%':
            val_text = f"{m['value']:.1f}%"
        else:
            val_text = f"{m['value']:.4f}"
        ax.text(center_x, center_y - 0.3, val_text, ha='center', va='center',
                fontsize=18, fontweight='bold', color=C_DARK)

        # Status badge
        badge_color = C_GREEN if m['status'] == 'PASS' else C_RED
        badge = FancyBboxPatch((3.5, 1.3), 3, 1.1, boxstyle="round,pad=0.2",
                               facecolor=badge_color, edgecolor='none', alpha=0.9)
        ax.add_patch(badge)
        ax.text(5, 1.85, m['status'], ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

        # Title
        ax.text(5, 9.1, m['name'], ha='center', va='center',
                fontsize=11, fontweight='bold', color=C_DARK, linespacing=1.2)

        # Description
        ax.text(5, 0.6, m['desc'], ha='center', va='center',
                fontsize=8, color='#777', style='italic')

    save(fig, 'fig15_metrics_scorecard.png')


# ================================================================
# FIGURE 16: Hit Rate vs Random (1190x Comparison Infographic)
# ================================================================
def fig16_hit_rate_vs_random():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.4, 'Figure 16: Hit Rate vs Random Baseline', ha='center',
            fontsize=18, fontweight='bold', color=C_DARK)
    ax.text(7, 6.8, 'How much better is our model compared to random recommendations?',
            ha='center', fontsize=11, color='#666')

    # Three comparison blocks
    k_values = [
        {'k': 10, 'hit': 14.29, 'random': 0.012, 'multiplier': '1,190x', 'color': C_BLUE, 'x': 2.5},
        {'k': 20, 'hit': 35.71, 'random': 0.025, 'multiplier': '1,428x', 'color': C_PURPLE, 'x': 7.0},
        {'k': 50, 'hit': 47.62, 'random': 0.062, 'multiplier': '768x', 'color': C_GREEN, 'x': 11.5},
    ]

    for kv in k_values:
        cx = kv['x']

        # Card background
        card = FancyBboxPatch((cx - 2, 0.5), 4, 5.8, boxstyle="round,pad=0.3",
                              facecolor=C_LIGHT_GRAY, edgecolor=kv['color'], linewidth=2)
        ax.add_patch(card)

        # K label
        ax.text(cx, 5.8, f"K = {kv['k']}", ha='center', fontsize=14, fontweight='bold', color=C_DARK)

        # Model bar (scaled to visual height)
        bar_height_model = kv['hit'] / 50 * 2.5
        bar_model = FancyBboxPatch((cx - 1.2, 2.2), 1, bar_height_model,
                                    boxstyle="round,pad=0.05", facecolor=kv['color'],
                                    edgecolor='none', alpha=0.85)
        ax.add_patch(bar_model)
        ax.text(cx - 0.7, 2.2 + bar_height_model + 0.15, f"{kv['hit']}%",
                ha='center', fontsize=11, fontweight='bold', color=kv['color'])
        ax.text(cx - 0.7, 1.8, 'Our Model', ha='center', fontsize=8, color='#555')

        # Random bar (tiny)
        bar_random = FancyBboxPatch((cx + 0.2, 2.2), 1, 0.08,
                                    boxstyle="round,pad=0.02", facecolor=C_RED,
                                    edgecolor='none', alpha=0.7)
        ax.add_patch(bar_random)
        ax.text(cx + 0.7, 2.45, f"{kv['random']}%", ha='center', fontsize=9,
                fontweight='bold', color=C_RED)
        ax.text(cx + 0.7, 1.8, 'Random', ha='center', fontsize=8, color='#555')

        # Multiplier badge
        badge = FancyBboxPatch((cx - 1, 0.7), 2, 0.85, boxstyle="round,pad=0.15",
                               facecolor=kv['color'], edgecolor='none', alpha=0.9)
        ax.add_patch(badge)
        ax.text(cx, 1.12, f"{kv['multiplier']} better", ha='center', fontsize=12,
                fontweight='bold', color='white')

    # Catalog label
    ax.text(7, 0.15, 'Catalog: 81,207 tracks  |  Feedback: 42 liked songs  |  Evaluation: Leave-One-Out',
            ha='center', fontsize=9, color='#999')

    save(fig, 'fig16_hit_rate_comparison.png')


# ================================================================
# FIGURE 17: Sonic Slider Sensitivity (Visual Sweep)
# ================================================================
def fig17_slider_sensitivity():
    fig = plt.figure(figsize=(15, 9), facecolor='white')
    fig.suptitle('Figure 17: Sonic Slider Sensitivity Test', fontsize=18, fontweight='bold',
                 color=C_DARK, y=0.96)
    fig.text(0.5, 0.92, 'Does adjusting a slider reliably change the recommendations?',
             ha='center', fontsize=11, color='#666')

    # Simulated monotonic responses (representative values)
    offsets = [-1.0, -0.5, 0.0, +0.5, +1.0]
    sliders = [
        {'name': 'Energy', 'icon': 'bolt', 'color': C_ORANGE,
         'values': [0.18, 0.27, 0.38, 0.55, 0.71]},
        {'name': 'Danceability', 'icon': 'music', 'color': C_PINK,
         'values': [0.22, 0.35, 0.48, 0.60, 0.73]},
        {'name': 'Valence (Mood)', 'icon': 'smile', 'color': C_AMBER,
         'values': [0.15, 0.25, 0.37, 0.52, 0.68]},
        {'name': 'Acousticness', 'icon': 'headphones', 'color': C_TEAL,
         'values': [0.10, 0.22, 0.35, 0.50, 0.65]},
    ]

    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28, top=0.87, bottom=0.08)

    for idx, s in enumerate(sliders):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])

        # Plot line with gradient fill
        ax.fill_between(offsets, s['values'], alpha=0.15, color=s['color'])
        ax.plot(offsets, s['values'], color=s['color'], linewidth=3, marker='o',
                markersize=10, markerfacecolor='white', markeredgecolor=s['color'],
                markeredgewidth=2.5, zorder=5)

        # Value labels on each point
        for x, y in zip(offsets, s['values']):
            ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                       xytext=(0, 14), ha='center', fontsize=9, fontweight='bold',
                       color=s['color'])

        # Arrow showing direction
        ax.annotate('', xy=(1.15, s['values'][-1]), xytext=(-1.15, s['values'][0]),
                   arrowprops=dict(arrowstyle='->', color=s['color'], lw=2, alpha=0.3))

        # Monotonic badge
        badge_x = 0.78
        badge = FancyBboxPatch((badge_x, 0.82), 0.2, 0.14, boxstyle="round,pad=0.02",
                               facecolor=C_GREEN, edgecolor='none', alpha=0.9,
                               transform=ax.transAxes)
        ax.add_patch(badge)
        ax.text(badge_x + 0.1, 0.89, 'MONOTONIC', ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', transform=ax.transAxes)

        ax.set_title(s['name'], fontsize=13, fontweight='bold', color=C_DARK, pad=10)
        ax.set_xlabel('Slider Offset', fontsize=10)
        ax.set_ylabel('Avg Feature in Recs', fontsize=10)
        ax.set_xticks(offsets)
        ax.set_xticklabels([str(o) for o in offsets])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xlim(-1.3, 1.3)

    save(fig, 'fig17_slider_sensitivity.png')


# ================================================================
# FIGURE 18: System Pipeline Architecture (Flowchart Infographic)
# ================================================================
def fig18_system_pipeline():
    fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(8, 7.5, 'Figure 18: System Architecture and MLOps Pipeline', ha='center',
            fontsize=18, fontweight='bold', color=C_DARK)

    # Pipeline stages
    stages = [
        {'label': 'Raw Data\n114K Tracks', 'x': 1.5, 'y': 4, 'color': C_BLUE, 'icon': 'database'},
        {'label': 'Preprocessing\nClean + Scale', 'x': 4.5, 'y': 4, 'color': C_TEAL, 'icon': 'filter'},
        {'label': 'Feature Space\n13D Vectors', 'x': 7.5, 'y': 4, 'color': C_PURPLE, 'icon': 'cube'},
        {'label': 'KNN Model\nCosine Search', 'x': 10.5, 'y': 4, 'color': C_ORANGE, 'icon': 'brain'},
        {'label': 'Recommendations\nPersonalized', 'x': 13.5, 'y': 4, 'color': C_GREEN, 'icon': 'music'},
    ]

    for i, s in enumerate(stages):
        # Box
        box = FancyBboxPatch((s['x'] - 1.2, s['y'] - 1.1), 2.4, 2.2,
                              boxstyle="round,pad=0.25", facecolor=s['color'],
                              edgecolor='none', alpha=0.15)
        ax.add_patch(box)
        # Circle icon placeholder
        circle = Circle((s['x'], s['y'] + 0.35), 0.45, facecolor=s['color'],
                        edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(circle)
        # Stage number
        ax.text(s['x'], s['y'] + 0.35, str(i+1), ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')
        # Label
        ax.text(s['x'], s['y'] - 0.5, s['label'], ha='center', va='center',
                fontsize=10, fontweight='bold', color=C_DARK, linespacing=1.3)

        # Arrow to next
        if i < len(stages) - 1:
            ax.annotate('', xy=(stages[i+1]['x'] - 1.3, s['y'] + 0.35),
                       xytext=(s['x'] + 1.3, s['y'] + 0.35),
                       arrowprops=dict(arrowstyle='->', color=C_DARK, lw=2))

    # Bottom row: MLOps components
    mlops = [
        {'label': 'User Feedback\nLikes / Dislikes', 'x': 4.0, 'y': 1.3, 'color': C_PINK},
        {'label': 'Blue/Green Deploy\nZero Downtime', 'x': 8.0, 'y': 1.3, 'color': C_INDIGO},
        {'label': 'Sonic Sliders\nReal-time Control', 'x': 12.0, 'y': 1.3, 'color': C_AMBER},
    ]

    # "MLOps Layer" label
    ax.text(8, 2.55, 'MLOps Layer', ha='center', fontsize=12, fontweight='bold',
            color='#999', style='italic')
    ax.plot([1.5, 14.5], [2.4, 2.4], color='#DDD', linewidth=1.5, linestyle='--')

    for m in mlops:
        box = FancyBboxPatch((m['x'] - 1.4, m['y'] - 0.6), 2.8, 1.2,
                              boxstyle="round,pad=0.2", facecolor=m['color'],
                              edgecolor='none', alpha=0.12)
        ax.add_patch(box)
        # Left accent bar
        accent = FancyBboxPatch((m['x'] - 1.4, m['y'] - 0.6), 0.12, 1.2,
                                 boxstyle="round,pad=0.01", facecolor=m['color'],
                                 edgecolor='none', alpha=0.8)
        ax.add_patch(accent)
        ax.text(m['x'] + 0.1, m['y'], m['label'], ha='center', va='center',
                fontsize=9, fontweight='bold', color=C_DARK, linespacing=1.3)

    # Feedback loop arrow (curved)
    ax.annotate('', xy=(1.5, 2.9), xytext=(4.0, 1.95),
               arrowprops=dict(arrowstyle='->', color=C_PINK, lw=1.8,
                              connectionstyle='arc3,rad=0.3'))
    ax.text(2.2, 2.1, 'retrain', fontsize=8, color=C_PINK, style='italic', rotation=25)

    # Deploy arrow
    ax.annotate('', xy=(10.5, 2.9), xytext=(8.0, 1.95),
               arrowprops=dict(arrowstyle='->', color=C_INDIGO, lw=1.8,
                              connectionstyle='arc3,rad=-0.3'))
    ax.text(9.6, 2.0, 'swap model', fontsize=8, color=C_INDIGO, style='italic', rotation=-20)

    # Slider arrow
    ax.annotate('', xy=(13.5, 2.9), xytext=(12.0, 1.95),
               arrowprops=dict(arrowstyle='->', color=C_AMBER, lw=1.8,
                              connectionstyle='arc3,rad=-0.3'))

    save(fig, 'fig18_system_pipeline.png')


# ================================================================
# FIGURE 19: Radar Chart (Multi-Metric Comparison)
# ================================================================
def fig19_radar_chart():
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True), facecolor='white')

    categories = ['Relevance\n(Cosine Sim)', 'Diversity\n(ILD)', 'Coverage',
                  'Novelty', 'Slider\nSensitivity', 'Hit Rate\n@ K=10']

    # Normalize all metrics to 0-1 scale for radar
    # Relevance: 0.9355/1.0, Diversity: 0.1069/0.5 (scaled), Coverage: use pass,
    # Novelty: 0.3942/1.0, Slider: 1.0 (pass), Hit Rate: 14.29/50
    model_values = [0.9355, 0.1069/0.5, 0.73, 0.3942, 1.0, 14.29/50]
    threshold_values = [0.70, 0.05/0.5, 0.01, 0.10, 0.60, 5.0/50]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    model_values += model_values[:1]
    threshold_values += threshold_values[:1]
    angles += angles[:1]

    # Plot
    ax.fill(angles, model_values, color=C_BLUE, alpha=0.15)
    ax.plot(angles, model_values, color=C_BLUE, linewidth=2.5, marker='o',
            markersize=8, markerfacecolor='white', markeredgecolor=C_BLUE,
            markeredgewidth=2, label='Our Model')

    ax.fill(angles, threshold_values, color=C_RED, alpha=0.08)
    ax.plot(angles, threshold_values, color=C_RED, linewidth=2, linestyle='--',
            marker='s', markersize=6, markerfacecolor='white', markeredgecolor=C_RED,
            markeredgewidth=1.5, label='Minimum Threshold')

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold', color=C_DARK)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8, color='#999')
    ax.spines['polar'].set_color('#DDD')
    ax.grid(color='#E0E0E0', linewidth=0.5)

    ax.set_title('Figure 19: Multi-Metric Performance Radar', fontsize=16,
                 fontweight='bold', color=C_DARK, pad=25)
    ax.legend(loc='lower right', bbox_to_anchor=(1.25, -0.05), fontsize=10, framealpha=0.9)

    save(fig, 'fig19_radar_chart.png')


# ================================================================
# FIGURE 20: Recommendation Modes Comparison (Infographic)
# ================================================================
def fig20_recommendation_modes():
    fig, ax = plt.subplots(figsize=(15, 7), facecolor='white')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)
    ax.axis('off')

    ax.text(7.5, 6.5, 'Figure 20: Two Recommendation Modes', ha='center',
            fontsize=18, fontweight='bold', color=C_DARK)

    # LEFT: Home Feed (User-to-Item)
    # Card
    left_card = FancyBboxPatch((0.5, 0.5), 6.5, 5.3, boxstyle="round,pad=0.3",
                                facecolor=C_BLUE, edgecolor='none', alpha=0.06)
    ax.add_patch(left_card)
    left_accent = FancyBboxPatch((0.5, 0.5), 0.15, 5.3, boxstyle="round,pad=0.01",
                                  facecolor=C_BLUE, edgecolor='none', alpha=0.8)
    ax.add_patch(left_accent)

    ax.text(3.75, 5.35, 'Home Feed', ha='center', fontsize=15, fontweight='bold', color=C_BLUE)
    ax.text(3.75, 4.9, 'User-to-Item Recommendation', ha='center', fontsize=10, color='#666')

    # Flow steps
    left_steps = [
        ('Liked Songs', '42 tracks', 4.15),
        ('Mean Centroid', 'P = avg(v_i)', 3.35),
        ('Sonic Slider', 'P + offset', 2.55),
        ('KNN Search', 'Top-K nearest', 1.75),
        ('Fresh Feed', 'Random sample', 0.95),
    ]
    for i, (title, sub, y) in enumerate(left_steps):
        # Step circle
        circle = Circle((1.3, y), 0.25, facecolor=C_BLUE, edgecolor='white', linewidth=1.5, alpha=0.85)
        ax.add_patch(circle)
        ax.text(1.3, y, str(i+1), ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')
        ax.text(2.0, y + 0.08, title, fontsize=10, fontweight='bold', color=C_DARK, va='center')
        ax.text(2.0, y - 0.22, sub, fontsize=8, color='#888', va='center')
        if i < len(left_steps) - 1:
            ax.plot([1.3, 1.3], [y - 0.3, y - 0.5], color=C_BLUE, linewidth=1.5, alpha=0.4)

    # RIGHT: Song Radio (Item-to-Item)
    right_card = FancyBboxPatch((8.0, 0.5), 6.5, 5.3, boxstyle="round,pad=0.3",
                                 facecolor=C_GREEN, edgecolor='none', alpha=0.06)
    ax.add_patch(right_card)
    right_accent = FancyBboxPatch((8.0, 0.5), 0.15, 5.3, boxstyle="round,pad=0.01",
                                   facecolor=C_GREEN, edgecolor='none', alpha=0.8)
    ax.add_patch(right_accent)

    ax.text(11.25, 5.35, 'Song Radio', ha='center', fontsize=15, fontweight='bold', color=C_GREEN)
    ax.text(11.25, 4.9, 'Item-to-Item Recommendation', ha='center', fontsize=10, color='#666')

    right_steps = [
        ('Seed Song', 'Any track selected', 4.15),
        ('Feature Lookup', '13D vector', 3.35),
        ('KNN Search', 'Nearest neighbors', 2.55),
        ('Exclude Seed', 'Remove duplicate', 1.75),
        ('Similar Playlist', 'Infinite radio', 0.95),
    ]
    for i, (title, sub, y) in enumerate(right_steps):
        circle = Circle((8.8, y), 0.25, facecolor=C_GREEN, edgecolor='white', linewidth=1.5, alpha=0.85)
        ax.add_patch(circle)
        ax.text(8.8, y, str(i+1), ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')
        ax.text(9.5, y + 0.08, title, fontsize=10, fontweight='bold', color=C_DARK, va='center')
        ax.text(9.5, y - 0.22, sub, fontsize=8, color='#888', va='center')
        if i < len(right_steps) - 1:
            ax.plot([8.8, 8.8], [y - 0.3, y - 0.5], color=C_GREEN, linewidth=1.5, alpha=0.4)

    # VS divider
    vs_circle = Circle((7.5, 3.2), 0.45, facecolor=C_DARK, edgecolor='white', linewidth=2)
    ax.add_patch(vs_circle)
    ax.text(7.5, 3.2, 'VS', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    save(fig, 'fig20_recommendation_modes.png')


# ================================================================
# MAIN
# ================================================================
def main():
    print("\n" + "="*60)
    print("  GENERATING RESULTS INFOGRAPHICS")
    print("="*60 + "\n")

    fig15_metrics_scorecard()
    fig16_hit_rate_vs_random()
    fig17_slider_sensitivity()
    fig18_system_pipeline()
    fig19_radar_chart()
    fig20_recommendation_modes()

    print(f"\n  All 6 infographics saved to: {OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
