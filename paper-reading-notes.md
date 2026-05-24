# [ROCODE: Integrating Backtracking Mechanism and Program Analysis in Large Language Models for Code Generation](https://arxiv.org/abs/2411.07112)

### Key Issues
- Where to roll back
- Where to roll back to
- How to avoid previous errors
## Key steps
### Incremental Error Detection
Statement $s_i = \mathcal{M}(x,S,S_{:i-1})$

Error $e_i = \mathcal{C}(S_{:i-1}||s_i) = \{\text{result, type, lineno, offset}\}$

### Strategic Rollback
Rollback point $r_e = [e.\text{lineno}, e.\text{offset}]$

Entropy at position $t$: $H_t = -\sum_{j=1}^{|V|} p(y_t=v_j|y_{<t},x)\log p(y_t=v_j|y_{<t},x)$

Rollback to highest entropy position: 

$t^* = \argmax_{t\in[0,|y|]}H_t$

$r_h = [\text{ConvertToLineno}(t^*,y)0]$

### Constraint Regeneration

Exponentially decaying penalty: $PN(v|y_{<t}) = 
\begin{cases}
    \lambda^{t-r}, \text{if } v=y_t, \\
    1, \text{otherwise}
\end{cases}$

# [SemGuard: Real-Time Semantic Evaluator for Correcting LLM-Generated Code](https://arxiv.org/abs/2509.24507v1)

## Limitations of ROCODE
- Post-generation semantic detection
- Imprecise backtracking-point location

## Approach
- Lightweight semantic evaluator on partial code

## Challenge
- Scarcity of line-level semantic annotations
- Undecidable semantic validity of partial code

## Data pipeline: *SemDiff*
