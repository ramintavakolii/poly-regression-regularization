import numpy as np
from matplotlib import pyplot as plt


class PolyReg:

    # ----------------------------------------- constructor for the class ------------------------------------------
    def __init__(self, coeffs=None, degree=0):
        # Initialize with coefficients or create zero coefficients for given degree
        if coeffs is None:
            coeffs = np.zeros(degree + 1)
            # print(self)
        self.set_coeffs(coeffs)

    # ----------------------------------------------- attributes --------------------------------------------------
    #       1.coeffs        2.nc (number of coefficients)        3.degree (represents the degree of the polynomial.)
    #       4.powers (an array of powers used to construct the design matrix.)

    # ------------------------------------------------- Methods ----------------------------------------------------

    def set_coeffs(self, coeffs):
        # Set the coefficients and compute related attributes
        self.coeffs = coeffs
        self.nc = len(coeffs)
        self.degree = self.nc - 1
        self.powers = np.arange(self.nc)

    def design_matrix(self, x, normalize=False):
        # Create the design matrix for input x
        # Reshape your data using array.reshape(-1, 1) if your data has a single feature
        X = x.reshape(-1, 1) ** self.powers.reshape(1, -1)
        # (optional)
        if normalize:
            mu = np.abs(X).mean(axis=0, keepdims=True)
            X /= mu
            return X, mu.ravel()
        return X

    def create_polydata(self, n, min_val, max_val, sigma_noise, sorted=True, upsample=1):
        # n : Number of data points             x : Generated input data
        # y : Output data computed from the polynomial plus noise

        # Generate polynomial data with noise
        x = np.random.uniform(min_val, max_val, n * upsample)
        if sorted:
            x = np.sort(x)
        x = x[::upsample]
        X = self.design_matrix(x)
        y = X.dot(self.coeffs) + np.random.randn(n) * \
            sigma_noise  # y = X.coeff + noise
        return x, y

    def __call__(self, x):
        # reg(x) is equivalent to calling reg.__call__(x)
        # Evaluate the polynomial at input x
        X = self.design_matrix(x)
        return X.dot(self.coeffs)

    def predict(self, x):
        # Predict the output for input x
        x = np.array(x)
        # The expression self(x) calls the __call__ method
        return self(x)

    def error(self, x, y):
        # y: Actual output data
        # Compute the mean squared error
        return np.mean((self.predict(x) - y) ** 2)

    def fit(self, x, y, reg_coeff=0):
        # Fit the polynomial model to data with regularization
        X = self.design_matrix(x)

        #   If reg_coeff > 0, regularization is applied.
        if reg_coeff > 0:
            nr, nc = X.shape
            X = np.vstack((X, np.sqrt(reg_coeff) * np.eye(nc)))
            y = np.hstack((y, np.zeros(nc)))
        # Solves the least squares problem 𝑋⋅coeffs = 𝑦 to find the best-fitting coefficients.
        # [0]: Extracts the coefficients from the solution.
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        self.set_coeffs(coeffs)


def generate_or_load_data(coeffs, n, low, high, noise_sigma, load=False):
    reg = PolyReg(coeffs)
    if load:
        # Load previously saved data if available
        saved_data = np.load('store.npz')
        x = saved_data['x']
        y = saved_data['y']
    else:
        # Generate new polynomial data
        x, y = reg.create_polydata(n, low, high, noise_sigma, upsample=4)
        np.savez('store.npz', x=x, y=y)
    print('Number of coefficient for generating data = ', reg.nc)
    return x, y


def plot_data(x, y):

    plt.cla()
    plt.plot(x, y, 'ro')
    plt.xlabel(r'$x$ (input)', fontsize=18)
    plt.ylabel(r'$y$ (output)', fontsize=18)
    plt.title("generated data", fontsize=20)
    plt.savefig('data.png')
    plt.show()


# Fit and plot models for different regularization parameters
def fit_and_plot_models(x, y, x_rng, x_lim, y_lim, degree, pow_list):
    err_list = []
    temp = 0
    for p in pow_list:
        temp += 1

        reg_coeff = np.power(10.0, p)
        reg = PolyReg(degree=degree)

        if temp == 1:
            print('Number of coefficient for model = ', reg.nc)

        reg.fit(x, y, reg_coeff)
        y_model = reg(x_rng)

        plt.cla()
        plt.plot(x, y, 'ro')
        plt.xlabel(r'$x$ (input)', fontsize=18)
        plt.ylabel(r'$y$ (output)', fontsize=18)
        plt.title(r"gamma = 10e%d, error = %.2f" %
                  (p, reg.error(x, y)), fontsize=20)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.plot(x_rng, y_model, 'b-')
        plt.show()
        plt.savefig('fit_deg_%d.png' % degree)

        err_list.append(reg.error(x, y))
    return err_list


# Plot error vs regularization parameter
def plot_errors_vs_reg_param(pow_list, err_list):
    plt.cla()
    plt.plot(pow_list, err_list, 'go-')
    plt.xlabel(r"$p$ (polynomial degree)", fontsize=18)
    plt.ylabel(r"error", fontsize=18)
    plt.savefig('error_vs_reg_param.png')
    plt.show()


# Plot training and testing data split
def plot_train_test_split(x_train, y_train, x_test, y_test):
    plt.cla()
    plt.plot(x_train, y_train, 'bo', label='train data')
    plt.plot(x_test, y_test, 'go', label='test data')
    plt.xlabel(r"$x$ (input)", fontsize=18)
    plt.ylabel(r"$y$ (output)", fontsize=18)
    plt.legend(loc='best')
    plt.show()
    plt.savefig('test_train_split.png')


# Fit models and plot errors for different regularization parameters
def fit_and_evaluate_train_test(x_train, y_train, x_test, y_test, x_rng, x_lim, y_lim, pow_list):

    err_train_list = []
    err_test_list = []
    pow_list = np.arange(1, 9, 1)
    for p in pow_list:

        reg_coeff = 0
        degree = p
        reg = PolyReg(degree=degree)
        reg.fit(x_train, y_train, reg_coeff)

        err_train = reg.error(x_train, y_train)
        err_test = reg.error(x_test, y_test)
        err_train_list.append(err_train)
        err_test_list.append(err_test)

        plt.clf()
        plt.plot(x_train, y_train, 'bo', label='train data')
        plt.plot(x_test, y_test, 'go', label='test data')
        plt.xlabel(r"$x$ (input)", fontsize=18)
        plt.ylabel(r"$y$ (output)", fontsize=18)
        plt.title("p = %d, err_train = %.2f, err_test = %.2f" %
                  (degree, err_train, err_test), fontsize=20)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.plot(x_rng, reg(x_rng), 'r-')
        plt.legend(loc='best')
        plt.show()
        plt.savefig('test_train_fit_%d.png' % degree)

    return err_train_list, err_test_list


# Plot training and testing errors
def plot_errors(pow_list, err_train_list, err_test_list, filename):
    plt.cla()
    plt.plot(pow_list, err_train_list, 'bo-', label='train error')
    plt.plot(pow_list, err_test_list, 'go-', label='test error')
    plt.xlabel(r"$p$ (polynomial degree)", fontsize=18)
    plt.ylabel(r"error", fontsize=18)
    plt.legend(loc='best')
    plt.ylim([0, 30])
    plt.show()
    plt.savefig('train_test_error.png')


def plot_log_errors(pow_list, err_train_list, err_test_list, filename):
    plt.cla()
    plt.plot(pow_list[:-1], np.log(err_train_list[:-1]),
             'bo-', label='log(train error)')
    plt.plot(pow_list[:-1], np.log(err_test_list[:-1]),
             'go-', label='log(test error)')
    plt.xlabel(r"$p$ (polynomial degree)", fontsize=18)
    plt.ylabel(r"log-error", fontsize=18)
    plt.legend(loc='best')
    plt.show()
    plt.savefig('train_test_error_log.png')


def test3():
    coeffs = np.array([-6, 11 * 4, -6 * 4**2, 1 * 4**3])
    n = 18
    low, high = 0.0, 1.0
    noise_sigma = 0.3
    load = False

    x, y = generate_or_load_data(coeffs, n, low, high, noise_sigma, load)

    # Split data into training and testing sets
    x_train, y_train = x[::2], y[::2]
    x_test, y_test = x[1::2], y[1::2]

    # Set plotting limits
    x_lim = [x.min() - 0.1 * (x.max() - x.min()),
             x.max() + 0.1 * (x.max() - x.min())]
    y_lim = [y.min() - 0.1 * (y.max() - y.min()),
             y.max() + 0.1 * (y.max() - y.min())]
    x_rng = np.linspace(*x_lim, 1000)

    # Plot and saving original data
    plot_data(x, y)

    pow_list = np.arange(-25, 10, 1)
    degree = n - 1
    err_list = fit_and_plot_models(x, y, x_rng, x_lim, y_lim, degree, pow_list)

    plot_errors_vs_reg_param(pow_list, err_list)
    plot_train_test_split(x_train, y_train, x_test, y_test)

    pow_list = np.arange(1, 9, 1)
    err_train_list, err_test_list = fit_and_evaluate_train_test(
        x_train, y_train, x_test, y_test, x_rng, x_lim, y_lim, pow_list)

    plot_errors(pow_list, err_train_list,
                err_test_list, "train_test_error.png")
    plot_log_errors(pow_list, err_train_list, err_test_list,
                    "train_test_error_log.png")


if __name__ == "__main__":
    test3()
