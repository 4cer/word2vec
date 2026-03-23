**Notation.** $V$ = vocab size, $N$ = embedding size, $C$ = context window count (number of context words fed in). $W_1 \in \mathbb{R}^{V \times N}$, $W_2 \in \mathbb{R}^{N \times V}$. One-hot input vectors $x_1, \dots, x_C \in \mathbb{R}^V$.
 
**Forward pass.**
 
$$h = \frac{1}{C} \sum_{i=1}^{C} W_1^T x_i$$
 
$$u = W_2^T h$$
 
$$\hat{y} = \text{softmax}(u), \quad \hat{y}_j = \frac{e^{u_j}}{\sum_k e^{u_k}}$$
 
$$\mathcal{L} = -\sum_j y_j \log \hat{y}_j \quad \text{(CCE, one-hot } y\text{)}$$
 
**1. Collapsed Softmax + CCE gradient**
 
Differentiating $\mathcal{L}$ through softmax gives the well-known clean result:
 
$$\frac{\partial \mathcal{L}}{\partial u} = \hat{y} - y$$
 
This skips computing the full Jacobian of softmax and is the gradient that enters backprop first.
 
**2. Gradient w.r.t. $W_2$**
 
$$\frac{\partial \mathcal{L}}{\partial W_2} = h \cdot \left(\hat{y} - y\right)^T$$
 
This is an outer product of the hidden vector and the output error.
 
**3. Gradient flowing back to $h$**
 
$$\frac{\partial \mathcal{L}}{\partial h} = W_2 \cdot \left(\hat{y} - y\right)$$
 
**4. Gradient w.r.t. $W_1$**
 
Because $h$ is an average of $C$ projected inputs, the gradient of $\mathcal{L}$ w.r.t. the hidden pre-average is divided equally across each context word's contribution. For each context word $x_i$:
 
$$\frac{\partial \mathcal{L}}{\partial W_1} \mathrel{+}= x_i \cdot \left(\frac{1}{C} \frac{\partial \mathcal{L}}{\partial h}\right)^T$$
 
Since each $x_i$ is one-hot, this reduces to adding $\frac{1}{C} \frac{\partial \mathcal{L}}{\partial h}$ into the row of $W_1$ corresponding to that context word index.
 
**SGD updates.**
 
$$W_2 \leftarrow W_2 - \eta \frac{\partial \mathcal{L}}{\partial W_2}, \qquad W_1 \leftarrow W_1 - \eta \frac{\partial \mathcal{L}}{\partial W_1}$$
 
**Note on collapsing Softmax + CCE.** The simplification $\partial \mathcal{L}/\partial u = \hat{y} - y$ is not specific to this model. It holds for any network whose final activation is softmax and whose loss is categorical cross-entropy. Because this pairing is extremely common, automatically detecting and collapsing it in the backprop graph (as done here via `COLLAPSE_TABLE`) is a sound general practice: it avoids computing the $V \times V$ softmax Jacobian and produces a numerically cleaner gradient.