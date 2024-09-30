

# Why does the ratio in PPO is dangerous when actions are long token sequences. 

Exploding or vanishing ratios. 

In PPO-KL, we try to optimize the following objective: 

$$
L^{\text{PPO-KL}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) - \beta \cdot \text{KL}(\pi_{\theta_{\text{old}}} || \pi_\theta) \right]
$$

Let $r_t(\theta) > 1 + \epsilon$. And let $\hat{A}_t = -1$. Then 

$$
\begin{aligned}
\min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t)\\
 = \text{min}(-r_t(\theta), - (\epsilon +1)) \\
= -r_t(\theta)
\end{aligned}
$$
And so the gradient step for the action at time $t$ is:

$$
\begin{aligned}
\nabla_\theta \log \pi_\theta(a_t | s_t) \left[ -r_t(\theta) - \beta \cdot \text{KL}(\pi_{\theta_{\text{old}}} || \pi_\theta) \right]
\end{aligned}
$$
If $\beta \cdot \text{KL}(\pi_{\theta_{\text{old}}} || \pi_\theta) \approx 0$, then the gradient step is:

$$
\begin{aligned}
\nabla_\theta \log \pi_\theta(a_t | s_t) \left[ -r_t(\theta) \right]
\end{aligned}
$$

Then the gradient update will have a large magnitude. 

Say the action at time $t$ is a sequence of tokens and for the sake of simplicity let us assume that the probability of each token is independant and constant. It is of $k_\theta$ for each token.
Then

$$
\begin{aligned}
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} =
\frac{\prod_{i=1}^{L} \pi_\theta(T_{i} | T_{l:i-1}, s_t)}{\prod_{i=1}^{L} \pi_{\theta_{\text{old}}}(T_{i} | T_{l:i-1}, s_t)}
=
\frac{\prod_{i=1}^{L} k_\theta}{\prod_{i=1}^{L} k_{\theta_{\text{old}}}}
=
(\frac{k_\theta}{k_{\theta_{\text{old}}}})^{L}
\end{aligned}
$$
which can explode or vanish with respect to $L$. 



