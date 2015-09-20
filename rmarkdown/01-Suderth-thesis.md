#### *Rene Welch* {.author}

-   [Basic stuff](#basic-stuff)
    -   [Exponential families](#exponential-families)
    -   [Entropy, information and
        divergence](#entropy-information-and-divergence)
    -   [Learning with priors](#learning-with-priors)
    -   [Examples of conjugate priors:](#examples-of-conjugate-priors)
-   [Graphical models](#graphical-models)

\\[ \\newcommand{\\E}{\\mathbb{E}} \\]

Basic stuff
===========

Exponential families
--------------------

The first concept are exponential families, which are families with a
known form and their value is characterized by sufficient statistics:

\\[ p(x|\\theta) = \\eta(x)\\exp\\left\\{ \\sum\_a \\theta\_a
\\phi\_a(x) - \\Phi(\\theta)\\right\\} \\]

In the machine learning setting the functions \\(\\phi\_a\\) are known
as *potentials* and the function \\(\\Phi\\) is defined so the density
function integrates one.

The parameter space is defined as \\[ \\Theta = \\left\\{\\theta :
|\\Phi(\\theta)| \< \\infty \\right\\} \\]

The *minimal representation* is made in such a way that the potentials
are constant. A couple of important results on the *potentials* are:

-   \\(\\frac{\\partial \\Phi(\\theta)}{\\partial \\theta\_a} = \\E[
    \\phi\_a(x)]\\)

-   \\(\\frac{\\partial\^2 \\Phi(\\theta)}{\\partial \\theta\_a
    \\partial \\theta\_b} = \\E [\\phi\_a(x)\\phi\_b(x)] -
    \\E[\\phi\_a(x)] \\E [\\phi\_b(x)]\\)

Entropy, information and divergence
-----------------------------------

*Shannon’s entropy* is defined as \\[ H(p) = \\int p(x)\\log
p(x)d\\mu(x) \\]

this function is concave, continuous and maximal for uniform densities.
The *Kullback - Leibler* divergence is defined as:

\\[ D(p||q) = \\int p(x)\\log \\frac{p(x)}{q(x)}d\\mu(x) \\]

and an important application of this functions is the mutual info.
between two random variable \\(x\\) and \\(y\\):

\\[ I(p\_{xy}) = D(p\_{xy}|| p\_x p\_y) = \\int \\int p\_{xy}(x,y)\\log
\\frac{p\_{x,y}(x,y)}{p\_x(x)p\_y(y)}dy dx \\]

In particular, given a target density \\(\\tilde{p}(x)\\) and
\\(p\_\\theta\\) an exponential family, the approximating density that
minimizes \\(D(\\tilde{p}||p\_\\theta)\\) has canonical parameters
\\(\\hat{\\theta}\\) choosen to match the expected values of that
family’s sufficient statistics:

\\[ \\E\_{\\hat{\\theta}}[\\phi\_a(x)] = \\int
\\phi\_a(x)\\tilde{p}(x)dx \\]

(the proof of this result is a direct result of optimizing \\(f(\\theta)
= D(\\tilde{p} | p\_\\theta)\\) as a function of \\(\\theta\\))

Additionally if \\((X\_i)\_{i=1}\^n \\sim \\tilde{p}\\), then the MLE
\\(\\hat{\\theta}\\) of the canonical parameters conincides with the
projection defined above:

\\[ \\hat{\\theta} = \\arg \\max\_\\theta \\sum\_i \\log p(x\_i |
\\theta) = \\arg \\min\_\\theta D(\\tilde{p} || p\_\\theta) \\]

Learning with priors
--------------------

To defined a *full bayesian model* we need the use of a prior
distribution, in the case of exponential fmailies the posterior
distribution is gonna have the form:

\\[ p(\\theta |\\lambda ) = \\exp\\left\\{ \\sum\_a \\theta\_a
\\lambda\_0 \\lambda\_a - \\lambda\_0 \\Phi(\\theta) - \\Omega
(\\lambda) \\right\\} \\]

*Proposition* If \\(X\_i \\sim p(x | \\theta )\\) (an exponential
family) with conjugate prior \\(p(\\theta |\\lambda)\\), then the
posterior parameters are updated by he rule:

\\[ p(\\theta | \\bf{x},\\lambda) = p(\\theta | \\lambda\^\*) \\]

where \\(\\lambda\^\*\_0 = \\lambda\_0 + N\\) and \\(\\lambda\_a\^\* =
\\frac{\\lambda\_0 \\lambda\_a + \\sum\_i \\phi\_a (x\_i)}{\\lambda\_0 +
N}\\)

Examples of conjugate priors:
-----------------------------

A couple of typical examples are:

1 - \\(X | \\theta\\) are multinomial and \$\$ is Dirichlet, a more
simple case of this example is when \\(K = 2\\), we have the Beta -
Binomial model

2 - \\(X|\\mu,\\Sigma\\) is normal and \\(\\mu\\) is normal and
\\(\\Sigma\\) is inverse Wishart. In the case of \\(X|\\mu,\\sigma\^2\\)
being univariate, then \\(\\sigma\^2\\) is inverse gamma, and the
conjugate is *t*

Graphical models
================

Hypergraphs \\(\\mathcal{H} = (\\mathcal{V},\\mathcal{F})\\) provide a
mean of describing probability distributions, in which case
