# User-level DP training

The idea of the LDP is to perturb the local data using the mechanism $\mathcal{M}$ such that the data perturbation is guaranteed to protect from inference attacks given parameters $\epsilon$ and $\delta$.

**1. We can define the (![formula](https://render.githubusercontent.com/render/math?math=\epsilon,\delta))-LDP as follows:**

A random mechanism ![formula](https://render.githubusercontent.com/render/math?math=\mathcal{M}) satisfies (![formula](https://render.githubusercontent.com/render/math?math=\epsilon,\delta))-LDP, where $\epsilon>0$ and $\delta\in[0, 1)$, if and only if for any two adjacent data sets $\mathcal{D}$ and $\mathcal{D'}$ in universe $\mathcal{X}$, we have:

$$
  ![formula](https://render.githubusercontent.com/render/math?math=Pr(\mathcal{M}(\mathcal{D}))\leq\;e^{\epsilon}Pr(\mathcal{M}(\mathcal{D'}))+\delta)
$$

**2. In this work, we select the Gaussian mechanism using $L_{2}$ norm sensitivity. In this setup, we perturb a sample output $s(x)$ by adding Gaussian noise with zero-mean and variance $\sigma^2\mathbf{I}$ for a given function $s(x)$ as below:**

$$
    \mathcal{M}(x) = s(x) + \mathcal{N}(0, \sigma^2\mathbf{I})
$$

**3. Formally, we define the sensitivity as the upper bound for the noise perturbation:**
$$
    \nabla s = \max_{\mathcal{D}, \mathcal{D'} \in \mathcal{X}} ||s(\mathcal{D})-s(\mathcal{D'})||_{2}
$$

In the setup of private FL, given two adjacent data sets $\mathcal{D}_{k}^{p}$ and $\mathcal{D'}_{k}^{p}$ and the model training process $\ell(\mathcal{D}_{k}^{p}, \theta^{t})$ in the $k^\text{th}$ client and $t^\text{th}$ global epoch, we can define the max sensitivity associated with this process as follows:

$$
  \nabla \ell = \max_{\mathcal{D}_{k}^{p}, \mathcal{D'}_{k}^{p} \in \mathcal{X}} ||\ell(\mathcal{D}_{k}^{p}, \theta^{t})-\ell(\mathcal{D'}_{k}^{p}, \theta^{t})||_{2}
$$

The above can be rewrite as:
$$
  \nabla \ell = \frac{\eta}{|\mathcal{D}_{k}^{p}|} \sum_{i=1}^{|\mathcal{D}_{k}^{p}|} \max_{\mathcal{D}_{k}^{p}, \mathcal{D'}_{k}^{p} \in \mathcal{X}} ||g^{t}_{k, i}(\mathcal{D}_{k, i}^{p})-g^{t}_{k, i}(\mathcal{D'}_{k, i}^{p})||
$$

**4. Finally, we can determine $\sigma_{k}$ of the Gaussian noise that satisfies $(\epsilon_{k}, \delta_{k})$-LDP for the $k^\text{th}$ client using the equation below:**

$$
  \sigma_{k} = \frac{\nabla \ell \sqrt{2qT\ln{(1/\delta_{k})}}}{\epsilon_{k}}
$$

**Finally, The UDP algorithm is shown below:**

![Alt text](../img/udp_algorithm.png?raw=true "UDP Federated Learning")

## If you find our work useful, you should also cite (and learn) these works  

**[UDP](https://arxiv.org/pdf/2003.00229.pdf)**

```
@article{wei2021user,
  title={User-level privacy-preserving federated learning: Analysis and performance optimization},
  author={Wei, Kang and Li, Jun and Ding, Ming and Ma, Chuan and Su, Hang and Zhang, Bo and Poor, H Vincent},
  journal={IEEE Transactions on Mobile Computing},
  year={2021},
  publisher={IEEE}
}
```

**[Deep learning withh DP](https://arxiv.org/pdf/2003.00229.pdf)**

```
@inproceedings{abadi2016deep,
  title={Deep learning with differential privacy},
  author={Abadi, Martin and Chu, Andy and Goodfellow, Ian and McMahan, H Brendan and Mironov, Ilya and Talwar, Kunal and Zhang, Li},
  booktitle={Proceedings of the 2016 ACM SIGSAC conference on computer and communications security},
  pages={308--318},
  year={2016}
}
```
