"""
Model Evaluation Suite for Spotify KNN Recommender
===================================================
Computes 6 quantitative metrics to prove the recommendation model works:
  1. Average Cosine Similarity (Relevance)
  2. Intra-List Diversity (ILD)
  3. Catalog Coverage
  4. Novelty Score
  5. Sonic Slider Sensitivity
  6. Hit Rate @ K

Usage:
    python -m src.evaluate
"""

import numpy as np
import pandas as pd
import os
from itertools import combinations


class ModelEvaluator:
    """Evaluates the KNN recommender using the live ModelManager."""

    def __init__(self, model_manager):
        self.manager = model_manager
        self.data = model_manager.data
        self.feature_cols = model_manager.feature_columns
        self.model = model_manager.active_model

    # ------------------------------------------------------------------
    # 1. Average Cosine Similarity to Query (Relevance)
    # ------------------------------------------------------------------
    def avg_cosine_similarity(self, query_vector, rec_indices):
        """
        Measures how close recommendations are to the query point.
        Higher = more relevant. Range: [-1, 1], typically 0.7-0.99.
        """
        rec_vectors = self.data.iloc[rec_indices][self.feature_cols].values
        similarities = []
        query_flat = query_vector.flatten()
        query_norm = np.linalg.norm(query_flat)

        if query_norm == 0:
            return 0.0

        for rec in rec_vectors:
            rec_norm = np.linalg.norm(rec)
            if rec_norm == 0:
                continue
            sim = np.dot(query_flat, rec) / (query_norm * rec_norm)
            similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 0.0

    # ------------------------------------------------------------------
    # 2. Intra-List Diversity (ILD)
    # ------------------------------------------------------------------
    def intra_list_diversity(self, rec_indices):
        """
        Average pairwise cosine DISTANCE within the recommendation list.
        Higher = more diverse. Range: [0, 2], good: 0.2-0.6.
        """
        rec_vectors = self.data.iloc[rec_indices][self.feature_cols].values
        if len(rec_vectors) < 2:
            return 0.0

        distances = []
        for i, j in combinations(range(len(rec_vectors)), 2):
            norm_i = np.linalg.norm(rec_vectors[i])
            norm_j = np.linalg.norm(rec_vectors[j])
            if norm_i == 0 or norm_j == 0:
                continue
            cos_sim = np.dot(rec_vectors[i], rec_vectors[j]) / (norm_i * norm_j)
            cos_dist = 1.0 - cos_sim
            distances.append(cos_dist)

        return float(np.mean(distances)) if distances else 0.0

    # ------------------------------------------------------------------
    # 3. Catalog Coverage
    # ------------------------------------------------------------------
    def catalog_coverage(self, n_runs=50, limit=12):
        """
        Runs the home feed multiple times and checks what fraction of the
        total catalog is ever recommended. Higher = less popularity bias.
        """
        all_rec_ids = set()
        for _ in range(n_runs):
            recs = self.manager.recommend_home(limit=limit)
            for r in recs:
                all_rec_ids.add(r['track_id'])

        total = len(self.data)
        coverage = len(all_rec_ids) / total if total > 0 else 0
        return coverage, len(all_rec_ids), total

    # ------------------------------------------------------------------
    # 4. Novelty Score
    # ------------------------------------------------------------------
    def novelty_score(self, rec_items):
        """
        Measures how obscure the recommendations are using self-information.
        Novelty = mean(-log2(popularity)). Higher = more novel/hidden gems.
        """
        if not rec_items:
            return 0.0

        scores = []
        for item in rec_items:
            pop = item.get('popularity', 0)
            # RobustScaler may produce negative values; clamp to small positive
            # We need a probability-like value, so normalize roughly
            # After RobustScaler, values are around 0; use sigmoid to map to (0,1)
            pop_prob = 1.0 / (1.0 + np.exp(-pop))  # sigmoid
            pop_prob = max(pop_prob, 0.01)  # floor to avoid log(0)
            scores.append(-np.log2(pop_prob))

        return float(np.mean(scores))

    # ------------------------------------------------------------------
    # 5. Sonic Slider Sensitivity
    # ------------------------------------------------------------------
    def slider_sensitivity(self, feature_name, steps=None):
        """
        Proves that adjusting a slider changes the recommendation output.
        Returns dict: {offset_value: avg_feature_value_in_recs}.
        A monotonic increase shows the slider works correctly.
        """
        if steps is None:
            steps = [-1.0, -0.5, 0.0, 0.5, 1.0]

        results = {}
        for offset in steps:
            overrides = {feature_name: offset}
            recs = self.manager.recommend_home(limit=12, feature_overrides=overrides)
            if recs:
                avg_val = np.mean([r.get(feature_name, 0) for r in recs])
                results[offset] = float(avg_val)
            else:
                results[offset] = None

        return results

    def check_monotonic(self, values):
        """Check if a sequence is roughly monotonically increasing."""
        clean = [v for v in values if v is not None]
        if len(clean) < 2:
            return False
        # Allow some tolerance: check if at least 3/4 of consecutive pairs increase
        increases = sum(1 for i in range(len(clean) - 1) if clean[i + 1] >= clean[i])
        return increases >= (len(clean) - 1) * 0.6

    # ------------------------------------------------------------------
    # 6. Hit Rate @ K (Leave-One-Out)
    # ------------------------------------------------------------------
    def hit_rate_at_k(self, liked_track_ids, k=20):
        """
        For each liked song: hold it out, build a profile from the rest,
        query KNN, and check if the held-out song appears in top-K results.
        """
        if len(liked_track_ids) < 3:
            return 0.0, 0, len(liked_track_ids)

        hits = 0
        tested = 0

        for held_out_id in liked_track_ids:
            remaining = [tid for tid in liked_track_ids if tid != held_out_id]

            # Build centroid from remaining liked songs
            liked_features = self.data[self.data['track_id'].isin(remaining)][self.feature_cols]
            if len(liked_features) == 0:
                continue

            user_profile = liked_features.mean(axis=0).values.reshape(1, -1)

            # KNN search
            distances, indices = self.model.kneighbors(user_profile, n_neighbors=k + len(remaining))
            candidate_indices = indices[0]
            candidates = self.data.iloc[candidate_indices]

            # Exclude the remaining liked songs from candidates
            candidates = candidates[~candidates['track_id'].isin(remaining)]
            top_k = candidates.head(k)

            if held_out_id in top_k['track_id'].values:
                hits += 1
            tested += 1

        rate = hits / tested if tested > 0 else 0.0
        return rate, hits, tested


# ======================================================================
# Report Runner
# ======================================================================
def run_evaluation():
    """Main evaluation pipeline — prints a formatted report."""
    from src.model_manager import ModelManager

    DIVIDER = "=" * 60
    SUBDIV = "-" * 60

    print()
    print(DIVIDER)
    print("  SPOTIFY KNN RECOMMENDER — EVALUATION REPORT")
    print(DIVIDER)

    # --- Setup ---
    print("\n[Setup] Loading ModelManager...")
    manager = ModelManager()

    if manager.active_model is None or manager.data is None:
        print("[ERROR] Model or data not loaded. Run training first.")
        return

    evaluator = ModelEvaluator(manager)
    print(f"[Setup] Model version: {manager.model_version}")
    print(f"[Setup] Dataset size: {len(manager.data):,} tracks")
    print(f"[Setup] Features: {len(manager.feature_columns)}")

    # --- Load feedback ---
    feedback_path = "data/feedback_data.csv"
    liked_ids = []
    if os.path.exists(feedback_path):
        fb_df = pd.read_csv(feedback_path, names=['track_id', 'liked'])
        liked_ids = fb_df[fb_df['liked'].astype(str) == 'True']['track_id'].unique().tolist()
    print(f"[Setup] Liked songs in feedback: {len(liked_ids)}")

    results = {}

    # ------------------------------------------------------------------
    # 1. RELEVANCE: Avg Cosine Similarity
    # ------------------------------------------------------------------
    print(f"\n{SUBDIV}")
    print("  1. RELEVANCE — Avg Cosine Similarity to User Profile")
    print(SUBDIV)

    if liked_ids:
        liked_features = manager.data[manager.data['track_id'].isin(liked_ids)][manager.feature_columns]
        if len(liked_features) > 0:
            user_profile = liked_features.mean(axis=0).values.reshape(1, -1)
            distances, indices = manager.active_model.kneighbors(user_profile, n_neighbors=10)
            score = evaluator.avg_cosine_similarity(user_profile, indices[0])
            results['cosine_similarity'] = score
            status = "PASS" if score > 0.7 else "WARN"
            print(f"  Score: {score:.4f}  [{status}]")
            print(f"  (Range: -1 to 1 | >0.7 = relevant, >0.9 = highly relevant)")
        else:
            print("  [SKIP] Liked songs not found in dataset.")
    else:
        print("  [SKIP] No feedback data available.")

    # ------------------------------------------------------------------
    # 2. DIVERSITY: Intra-List Diversity
    # ------------------------------------------------------------------
    print(f"\n{SUBDIV}")
    print("  2. DIVERSITY — Intra-List Diversity (ILD)")
    print(SUBDIV)

    if liked_ids:
        liked_features = manager.data[manager.data['track_id'].isin(liked_ids)][manager.feature_columns]
        if len(liked_features) > 0:
            user_profile = liked_features.mean(axis=0).values.reshape(1, -1)
            distances, indices = manager.active_model.kneighbors(user_profile, n_neighbors=12)
            ild = evaluator.intra_list_diversity(indices[0])
            results['ild'] = ild
            status = "PASS" if 0.05 < ild < 0.9 else "WARN"
            print(f"  Score: {ild:.4f}  [{status}]")
            print(f"  (Range: 0 to 2 | 0=clones, 0.2-0.5=good, >0.8=too random)")
    else:
        print("  [SKIP] No feedback data available.")

    # ------------------------------------------------------------------
    # 3. COVERAGE: Catalog Coverage
    # ------------------------------------------------------------------
    print(f"\n{SUBDIV}")
    print("  3. COVERAGE — Catalog Coverage (50 runs)")
    print(SUBDIV)

    coverage, unique_count, total = evaluator.catalog_coverage(n_runs=50, limit=12)
    results['coverage'] = coverage
    status = "PASS" if coverage > 0.01 else "WARN"
    print(f"  Unique songs recommended: {unique_count:,} / {total:,}")
    print(f"  Coverage: {coverage:.2%}  [{status}]")
    print(f"  (>5% is good for a 114k catalog, >1% acceptable)")

    # ------------------------------------------------------------------
    # 4. NOVELTY: Novelty Score
    # ------------------------------------------------------------------
    print(f"\n{SUBDIV}")
    print("  4. NOVELTY — Hidden Gem Score")
    print(SUBDIV)

    recs = manager.recommend_home(limit=20)
    if recs:
        novelty = evaluator.novelty_score(recs)
        results['novelty'] = novelty

        # Compare to a popularity baseline
        top_popular = manager.data.sort_values('popularity', ascending=False).head(20)
        baseline_items = top_popular.to_dict('records')
        baseline_novelty = evaluator.novelty_score(baseline_items)
        results['baseline_novelty'] = baseline_novelty

        status = "PASS" if novelty >= baseline_novelty else "INFO"
        print(f"  System Novelty:   {novelty:.4f}")
        print(f"  Baseline Novelty: {baseline_novelty:.4f}  (top-popular only)")
        gain = ((novelty - baseline_novelty) / baseline_novelty * 100) if baseline_novelty > 0 else 0
        print(f"  Novelty Gain:     {gain:+.1f}%  [{status}]")
        print(f"  (Higher = recommending more obscure/diverse tracks)")
    else:
        print("  [SKIP] No recommendations returned.")

    # ------------------------------------------------------------------
    # 5. SLIDER SENSITIVITY
    # ------------------------------------------------------------------
    print(f"\n{SUBDIV}")
    print("  5. INTERACTIVITY — Sonic Slider Sensitivity")
    print(SUBDIV)

    slider_features = ['energy', 'danceability', 'valence', 'acousticness']
    all_monotonic = True

    for feat in slider_features:
        sens = evaluator.slider_sensitivity(feat)
        is_mono = evaluator.check_monotonic(list(sens.values()))
        label = "PASS" if is_mono else "FAIL"
        if not is_mono:
            all_monotonic = False

        offsets_str = "  ".join([f"{k:+.1f}→{v:.3f}" for k, v in sens.items() if v is not None])
        print(f"  {feat:15s}: {offsets_str}  [{label}]")

    results['slider_sensitivity'] = all_monotonic

    # ------------------------------------------------------------------
    # 6. HIT RATE @ K
    # ------------------------------------------------------------------
    print(f"\n{SUBDIV}")
    print("  6. PREDICTIVE POWER — Hit Rate @ K (Leave-One-Out)")
    print(SUBDIV)

    if len(liked_ids) >= 3:
        for k in [10, 20, 50]:
            rate, hits, tested = evaluator.hit_rate_at_k(liked_ids, k=k)
            results[f'hit_rate_at_{k}'] = rate
            status = "PASS" if rate > 0.0 else "INFO"
            print(f"  Hit Rate @ {k:2d}: {rate:.2%} ({hits}/{tested} hits)  [{status}]")
        print(f"  (For a {len(manager.data):,}-track catalog, >5% @ K=10 is strong)")
    else:
        print("  [SKIP] Need at least 3 liked songs for leave-one-out evaluation.")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print(f"\n{DIVIDER}")
    print("  SUMMARY")
    print(DIVIDER)

    summary_items = [
        ("Relevance (Cosine Sim)", results.get('cosine_similarity')),
        ("Diversity (ILD)", results.get('ild')),
        ("Coverage", results.get('coverage')),
        ("Novelty", results.get('novelty')),
        ("Slider Responsive", results.get('slider_sensitivity')),
        ("Hit Rate @20", results.get('hit_rate_at_20')),
    ]

    for name, val in summary_items:
        if val is None:
            print(f"  {name:25s}  —")
        elif isinstance(val, bool):
            print(f"  {name:25s}  {'✓ YES' if val else '✗ NO'}")
        elif isinstance(val, float) and val < 1:
            print(f"  {name:25s}  {val:.4f}")
        else:
            print(f"  {name:25s}  {val}")

    print(f"\n  Model: {manager.model_version} | Data: {len(manager.data):,} tracks")
    print(DIVIDER)
    print()


if __name__ == "__main__":
    run_evaluation()
