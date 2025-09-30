# FORMULA 1 POLE PREDICTION - STAGE-1 EVALUATION REPORT
================================================================================

## EXECUTIVE SUMMARY

**Best Model:** FP3_Baseline
**Best Top-1 Accuracy:** 0.039 (3.9%)
**Best MRR:** 0.653

## MODEL PERFORMANCE COMPARISON

### Train Set Performance

       model  top1_accuracy  top3_accuracy  top5_accuracy   mrr  ndcg_at_5
FP3_Baseline          0.039          0.812          0.938 0.653      0.725
         GBM          0.039          0.812          0.938 0.653      0.725

## DETAILED ANALYSIS

### Model vs Baseline Comparison

**FP3_Baseline:**
- Top-1 Accuracy: 0.039 (vs baseline: +0.039)
- Improvement: N/A

**GBM:**
- Top-1 Accuracy: 0.039 (vs baseline: +0.039)
- Improvement: N/A

## KEY INSIGHTS

- **Low pole prediction accuracy** - Room for significant improvement
- **Strong ranking performance** - Pole sitter often in top 3 predictions
- **Data Quality Issue:** Validation set contains no pole positions
- **Data Quality Issue:** Test set contains no pole positions

## RECOMMENDATIONS

1. **Fix data splits** - Ensure validation/test sets contain pole positions
2. **Feature engineering** - Add circuit-specific and weather features
3. **Advanced models** - Try neural networks and ensemble methods
4. **Calibration** - Implement probability calibration for confidence scores
