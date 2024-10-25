

# Problem with initial action:
$$
\begin{align}
\nabla_\theta \log( p(t_1, t_2, \ldots, t_n) )
= \nabla_\theta \log( \prod_{i=1}^n p(t_i|t_1, \ldots, t_{i-1}) ) \\
= \sum_{i=1}^n \nabla_\theta \log( p(t_i|t_1, \ldots, t_{i-1}) ) \\
\end{align}
$$
This is summed over all trajectories according to reward. But say some initial tokens are correctly always the same. For example, when the initial tokens must be `<finalize>`. Then there will be a term 
$\bar{R} \nabla_\theta \log ( p_\theta (\text{<finalize>}) )$. If $\bar{R}$ is negative this will give bad results: the model will learn to avoid saying `<finalize>` even though it is the correct thing to do. 