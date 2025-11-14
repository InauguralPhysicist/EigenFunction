# Loop Prevention Demo

## What This Shows

Standard attention can get stuck in loops on self-referential patterns. This demo compares two approaches:

1. **Standard Attention**: Dot-product similarity (self-similarity = 1.0)
2. **Spacetime Feedback**: Lorentz-invariant attention with feedback correction

## Running the Demo

```bash
python demo_loop_prevention.py
```

## Results

### Test 1: Self-Referential Input

**Standard attention:**
- Iterations: 3
- Loop detected: Yes (similarity > 0.99)

**Spacetime feedback:**
- Iterations: 1
- Converged: Yes
- Final imbalance: 0.089

### Test 2: Query Refinement

**Spacetime feedback:**
- Iterations: 1
- Converged: Yes
- Final imbalance: 0.023

## How It Works

The spacetime architecture uses three components:

- **Timelike branch**: Causal attention (sequential)
- **Spacelike branch**: Non-causal attention (parallel)
- **Lightlike monitor**: Computes spacetime interval ds² = ||spacelike||² - ||timelike||²

When ds² indicates imbalance, feedback correction is applied proportional to the magnitude.

## Files

- `demo_loop_prevention.py` - Demonstration code
- `standard_attention.py` - Baseline implementation
- `spacetime_feedback.py` - Spacetime architecture
- `test_spacetime_feedback.py` - Unit tests

## Technical Details

See `ARCHITECTURE.md` for mathematical formulation and `MATH_VERIFICATION.md` for proof that the spacetime interval computation is correct.
