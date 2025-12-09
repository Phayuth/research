import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import os


def gaussian_pdf_1d():
    xm = 0
    xstd = 0.3
    npoints = 500
    xsample = np.random.normal(xm, xstd, (npoints,))
    y = 0 * np.arange(npoints)

    def normal_pdf(x, mu, sigma):
        return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
            -0.5 * ((x - mu) / sigma) ** 2
        )

    def normal_pdf_grad(x, mu, sigma):
        f = normal_pdf(x, mu, sigma)
        return -(x - mu) / (sigma**2) * f

    xpdf = np.linspace(xm - 4 * xstd, xm + 4 * xstd, npoints)
    ygussian = normal_pdf(xpdf, xm, xstd)
    ypguassian = normal_pdf_grad(xpdf, xm, xstd)

    fig, ax = plt.subplots()
    ax.scatter(xsample, y, s=5, c="blue", label="Data points")
    ax.plot(xpdf, ygussian, c="red", label="Gaussian distribution PDF")
    ax.plot(xpdf, ypguassian, c="green", label="Gaussian PDF Gradient")
    ax.set_aspect("equal")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.legend()
    ax.set_xlim(xm - 4 * xstd, xm + 4 * xstd)
    ax.set_ylim(-0.1, 2)
    plt.show()


def gaussian_pdf_2d():
    mu = np.array([0, 0])  # Mean vector
    Sigma = np.array([[0.3, 0.2], [0.2, 0.7]])  # Covariance must positive definite
    npoints = 500
    xsample = np.random.multivariate_normal(mu, Sigma, npoints)
    x1 = np.linspace(
        mu[0] - 4 * np.sqrt(Sigma[0, 0]),
        mu[0] + 4 * np.sqrt(Sigma[0, 0]),
        100,
    )
    x2 = np.linspace(
        mu[1] - 4 * np.sqrt(Sigma[1, 1]),
        mu[1] + 4 * np.sqrt(Sigma[1, 1]),
        100,
    )
    X1, X2 = np.meshgrid(x1, x2)
    pos = np.dstack((X1, X2))

    Sigma_inv = np.linalg.inv(Sigma)
    Sigma_det = np.linalg.det(Sigma)
    norm_const = 1.0 / (2 * np.pi * np.sqrt(Sigma_det))
    diff = pos - mu  # shape (...,2)
    exponent = np.einsum("...i,ij,...j", diff, Sigma_inv, diff)  # shape (...,)
    Y = norm_const * np.exp(-0.5 * exponent)  # PDF values on grid, shape (...,)

    # ----------------------------
    # Jacobian (gradient) of Y:  ∇f(x) = - f(x) * Sigma^{-1} (x - mu)
    # Vectorized computation:
    # ----------------------------
    # tmp = Sigma^{-1} @ (x-mu) for each grid point -> shape (...,2)
    tmp = np.einsum("ij,...j->...i", Sigma_inv, diff)  # shape (...,2)

    # gradient arrays
    grad = -Y[..., np.newaxis] * tmp  # shape (...,2)
    dY_dx1 = grad[..., 0]
    dY_dx2 = grad[..., 1]

    x0 = np.array([0.1, -0.05])
    f_x0 = norm_const * np.exp(-0.5 * (x0 - mu) @ Sigma_inv @ (x0 - mu))
    grad_x0 = -f_x0 * (Sigma_inv @ (x0 - mu))
    print("f(x0) =", f_x0)
    print("grad f(x0) =", grad_x0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        xsample[:, 0],
        xsample[:, 1],
        zs=0,
        zdir="z",
        s=5,
        c="blue",
        label="Data points",
    )
    # step = 8
    # ax.quiver(
    #     X1[::step, ::step],
    #     X2[::step, ::step],
    #     dY_dx1[::step, ::step],
    #     dY_dx2[::step, ::step],
    #     scale=50,  # adjust scale for arrow length visibility
    #     width=0.003,
    #     alpha=0.8,
    #     color="red",
    #     label="Gradient (Jacobian)",
    # )
    ax.plot_surface(X1, X2, Y, cmap="viridis", alpha=0.7)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Probability Density")
    ax.set_title("2D Gaussian Distribution PDF")
    plt.show()


def guassian_gmm_1d():
    # Generate synthetic 1D data
    np.random.seed(42)
    x = np.concatenate(
        [np.random.normal(-2, 0.5, 300), np.random.normal(3, 1.0, 300)]
    )
    x = x.reshape(-1, 1)

    K = 2
    gmm = GaussianMixture(n_components=K, covariance_type="full")
    gmm.fit(x)

    # Extract learned parameters
    pi = gmm.weights_  # mixture weights
    mu = gmm.means_.flatten()  # means
    sigma = np.sqrt(gmm.covariances_.flatten())  # standard deviations

    print("Mixture weights:", pi)
    print("Means:", mu)
    print("Sigmas:", sigma)

    def normal_pdf(x, mu, sigma):
        return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
            -0.5 * ((x - mu) / sigma) ** 2
        )

    def gmm_pdf(x, pi, mu, sigma):
        total = np.zeros_like(x, dtype=float)
        for k in range(len(pi)):
            total += pi[k] * normal_pdf(x, mu[k], sigma[k])
        return total

    def gmm_pdf_grad(x, pi, mu, sigma):
        # dp/dx = sum_k pi_k * [- (x - mu_k) / sigma_k^2 ] * N(x | mu_k)
        grad = np.zeros_like(x, dtype=float)
        for k in range(len(pi)):
            pdf_k = normal_pdf(x, mu[k], sigma[k])
            grad += pi[k] * (-(x - mu[k]) / (sigma[k] ** 2)) * pdf_k
        return grad

    xgrid = np.linspace(x.min() - 1, x.max() + 1, 500).reshape(-1, 1)
    pdf = np.exp(gmm.score_samples(xgrid))  # full mixture PDF

    fig, ax = plt.subplots()
    ax.hist(x, bins=40, density=True, alpha=0.4, label="Data histogram")
    ax.plot(xgrid, pdf, linewidth=2, label="GMM PDF curve")
    ax.plot(
        xgrid,
        gmm_pdf_grad(xgrid, pi, mu, sigma),
        linewidth=2,
        label="GMM PDF Gradient",
    )

    # individual components
    for k in range(K):
        comp_pdf = (
            pi[k]
            * (1 / np.sqrt(2 * np.pi * sigma[k] ** 2))
            * np.exp(-((xgrid - mu[k]) ** 2) / (2 * sigma[k] ** 2))
        )
        ax.plot(xgrid, comp_pdf, linestyle="--", label=f"Component {k+1}")

    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()
    plt.show()


def guassian_gmm_1d_Kunkown():
    """
    Determine the optimal number of Gaussian components in a 1D GMM using AIC/BIC
    """
    random_state = np.random.RandomState(seed=1)

    X = np.concatenate(
        [
            random_state.normal(-1, 1.5, 350),
            random_state.normal(0, 1, 500),
            random_state.normal(3, 0.5, 150),
        ]
    ).reshape(-1, 1)

    # ------------------------------------------------------------
    # Learn the best-fit GaussianMixture models
    #  Here we'll use scikit-learn's GaussianMixture model. The fit() method
    #  uses an Expectation-Maximization approach to find the best
    #  mixture of Gaussians for the data

    # fit models with 1-10 components find the best number of components using AIC
    N = np.arange(1, 11)
    models = [None for i in range(len(N))]

    for i in range(len(N)):
        models[i] = GaussianMixture(N[i]).fit(X)

    # compute the AIC and the BIC
    AIC = [m.aic(X) for m in models]
    BIC = [m.bic(X) for m in models]

    # ------------------------------------------------------------
    # Plot the results
    #  We'll use three panels:
    #   1) data + best-fit mixture
    #   2) AIC and BIC vs number of components
    #   3) probability that a point came from each component

    fig = plt.figure(figsize=(5, 1.7))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.21, top=0.9, wspace=0.5)

    # plot 1: data + best-fit mixture
    ax = fig.add_subplot(131)
    M_best = models[np.argmin(AIC)]

    x = np.linspace(-6, 6, 1000)
    logprob = M_best.score_samples(x.reshape(-1, 1))
    responsibilities = M_best.predict_proba(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    ax.hist(X, 30, density=True, histtype="stepfilled", alpha=0.4)
    ax.plot(x, pdf, "-k")
    ax.plot(x, pdf_individual, "--k")
    ax.text(
        0.04, 0.96, "Best-fit Mixture", ha="left", va="top", transform=ax.transAxes
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$p(x)$")

    # plot 2: AIC and BIC
    ax = fig.add_subplot(132)
    ax.plot(N, AIC, "-k", label="AIC")
    ax.plot(N, BIC, "--k", label="BIC")
    ax.set_xlabel("n. components")
    ax.set_ylabel("information criterion")
    ax.legend(loc=2)

    # plot 3: posterior probabilities for each component
    ax = fig.add_subplot(133)

    p = responsibilities
    p = p[:, (1, 0, 2)]  # rearrange order so the plot looks better
    p = p.cumsum(1).T

    ax.fill_between(x, 0, p[0], color="gray", alpha=0.3)
    ax.fill_between(x, p[0], p[1], color="gray", alpha=0.5)
    ax.fill_between(x, p[1], 1, color="gray", alpha=0.7)
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$p({\rm class}|x)$")

    ax.text(-5, 0.3, "class 1", rotation="vertical")
    ax.text(0, 0.5, "class 2", rotation="vertical")
    ax.text(3, 0.3, "class 3", rotation="vertical")

    plt.show()


def guassian_gmm_2d():
    pass


if __name__ == "__main__":
    # gaussian_pdf_1d()
    # gaussian_pdf_2d()
    # guassian_gmm_1d()
    guassian_gmm_1d_Kunkown()
