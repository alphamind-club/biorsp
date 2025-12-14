### Method Overview

BioRSP identifies genes with directional expression gradients in 2D single-cell embeddings while controlling for technical confounders. The core statistic is Anisotropy Radial Index Analysis (ARIA), measured as the circular Wasserstein-1 distance between observed and expected angular distributions under covariate-conditional nulls.

#### Mathematical Derivation

Given cell positions $(x_i, y_i)$ and angles $\theta_i = \tan^{-1}((y_i - y_0)/(x_i - x_0))$ relative to vantage point $(x_0, y_0)$, we compute weighted angular histograms:

$$ n_F(\phi_k) = \sum_i w_i K(\theta_i - \phi_k) $$
$$ n_B(\phi_k) = \sum_i b_i K(\theta_i - \phi_k) $$

where $w_i$ is gene expression, $b_i$ is background weight, and $K$ is a smoothing kernel. Normalized densities are:

$$ p_F(\phi_k) = \frac{n_F(\phi_k)}{\sum_j n_F(\phi_j)}, \quad p_B(\phi_k) = \frac{n_B(\phi_k)}{\sum_j n_B(\phi_j)} $$

The RSP curve is: $RSP(\phi_k) = \log \frac{p_F(\phi_k)}{p_B(\phi_k)}$

ARIA measures the circular Wasserstein-1 distance between observed ($p_F$) and expected ($p_B$) angular distributions:

$$ ARIA = W_1(p_F, p_B) $$

where $W_1$ is the Wasserstein-1 distance on the circle, quantifying minimum "work" to transform distributions.

#### Statistical Testing

We use conditional permutation tests where null hypothesis is gene expression independence from spatial position conditional on covariates $Z$. Permutations preserve covariate distributions via stratified or kNN shuffling.

P-value: $p = \frac{1 + \sum_b I(ARIA^{(b)} \ge ARIA^{obs})}{1 + B}$

For large datasets, approximate Z-scores via moment-based estimation of null distribution.

### Complexity Analysis

- **Preprocessing:** $O(N \log N)$ for kNN graph
- **Gene-independent permutation indices:** $O(B N)$ stratified, $O(B N \log N)$ kNN
- **Per gene:** $O(N)$ for CRA computation, $O(B N)$ for permutation test
- **Total:** $O(G B N)$ with gene-independent indices providing 10–100× speedup

### API Overview

Main function: `biorsp.find_spatially_patterned_genes()`

**Key outputs:**

- `ARIA`: Primary effect size (Wasserstein distance)
- `Z_W1`: Standardized effect size
- `p_CRA`: P-value from permutation test
- `q_CRA`: FDR-adjusted q-value
- `discovery`: Boolean for significant genes

**Guardrails:** Minimum cell counts (200 pub/50 exploratory), vantage stability, embedding distortion checks.

### Memory Usage

Peak: Expression matrix ($N \times G$) + permutation indices ($B \times N$). For $N=50k, G=20k, B=1000$: ~500 MB expression + ~400 MB indices.
