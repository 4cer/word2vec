**Notation.** $V$ = vocab size, $N$ = embedding size, $C$ = context window count (number of context words fed in). $W_1 \in \mathbb{R}^{V \times N}$, $W_2 \in \mathbb{R}^{N \times V}$. One-hot input vectors $x_1, \dots, x_C \in \mathbb{R}^V$.

**Forward pass.**

$$h = \frac{1}{C} \sum_{i=1}^{C} W_1^T x_i$$

$$u = W_2^T h$$

$$\hat{y} = \text{softmax}(u), \quad \hat{y}_j = \frac{e^{u_j}}{\sum_k e^{u_k}}$$

$$\mathcal{L} = -\sum_j y_j \log \hat{y}_j \quad \text{(CCE, one-hot } y\text{)}$$

**1. Gradient of CCE w.r.t. softmax output $\hat{y}$**

Starting from the loss directly:

$$\mathcal{L} = -\sum_j y_j \log \hat{y}_j$$

Differentiating with respect to a single output element $\hat{y}_j$:

$$\frac{\partial \mathcal{L}}{\partial \hat{y}_j} = -\frac{y_j}{\hat{y}_j}$$

Since $y$ is one-hot with $y_c = 1$ for the correct class $c$ and zero elsewhere, only one term survives:

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}}$$

**2. Gradient of Softmax w.r.t. its input $u$**

For $\hat{y}_j = e^{u_j} / S$ where $S = \sum_k e^{u_k}$, differentiating with respect to $u_i$:

When $i = j$:

$$\frac{\partial \hat{y}_j}{\partial u_j} = \frac{e^{u_j} \cdot S - e^{u_j} \cdot e^{u_j}}{S^2} = \hat{y}_j(1 - \hat{y}_j)$$

When $i \neq j$:

$$\frac{\partial \hat{y}_j}{\partial u_i} = \frac{0 - e^{u_j} \cdot e^{u_i}}{S^2} = -\hat{y}_j \hat{y}_i$$

Compactly, the full Jacobian is:

$$\frac{\partial \hat{y}}{\partial u} = \text{diag}(\hat{y}) - \hat{y}\hat{y}^T \in \mathbb{R}^{V \times V}$$

**3. Chain rule collapse**

Applying the chain rule naively would require materialising the $V \times V$ Jacobian above:

$$\frac{\partial \mathcal{L}}{\partial u} = \frac{\partial \hat{y}}{\partial u}^T \cdot \frac{\partial \mathcal{L}}{\partial \hat{y}} = \left(\text{diag}(\hat{y}) - \hat{y}\hat{y}^T\right) \cdot \left(-\frac{y}{\hat{y}}\right)$$

Expanding element $j$:

$$\frac{\partial \mathcal{L}}{\partial u_j} = \hat{y}_j \left(-\frac{y_j}{\hat{y}_j}\right) - \hat{y}_j \sum_k \hat{y}_k \left(-\frac{y_k}{\hat{y}_k}\right) = -y_j + \hat{y}_j \sum_k y_k$$

Since $y$ is one-hot, $\sum_k y_k = 1$, giving:

$$\frac{\partial \mathcal{L}}{\partial u} = \hat{y} - y$$

This is the collapsed gradient used in backpropagation. It avoids forming the $V \times V$ Jacobian entirely and is numerically cleaner. This pairing recurs in any softmax + CCE network, so it is detected and applied automatically via `COLLAPSE_TABLE` in the optimizer.

**4. Gradient w.r.t. $W_2$**

$$\frac{\partial \mathcal{L}}{\partial W_2} = h \cdot \left(\hat{y} - y\right)^T$$

This is an outer product of the hidden vector and the output error.

**5. Gradient flowing back to $h$**

$$\frac{\partial \mathcal{L}}{\partial h} = W_2 \cdot \left(\hat{y} - y\right)$$

**6. Gradient w.r.t. $W_1$**

Because $h$ is an average of $C$ projected inputs, the gradient of $\mathcal{L}$ w.r.t. the hidden pre-average is divided equally across each context word's contribution. For each context word $x_i$:

$$\frac{\partial \mathcal{L}}{\partial W_1} \mathrel{+}= x_i \cdot \left(\frac{1}{C} \frac{\partial \mathcal{L}}{\partial h}\right)^T$$

Since each $x_i$ is one-hot, this reduces to adding $\frac{1}{C} \frac{\partial \mathcal{L}}{\partial h}$ into the row of $W_1$ corresponding to that context word index.

**SGD updates.**

$$W_2 \leftarrow W_2 - \eta \frac{\partial \mathcal{L}}{\partial W_2}, \qquad W_1 \leftarrow W_1 - \eta \frac{\partial \mathcal{L}}{\partial W_1}$$