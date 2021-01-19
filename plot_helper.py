import matplotlib.pyplot as plt

def plot_graph(tau, uu, lambdas, lIW):
    plt.plot(tau,abs(uu)**2)
    plt.title('Time Domain')
    plt.xlabel('Time tau (ps)')
    plt.ylabel('Power(W)')
    plt.show()

    plt.stem(lambdas,lIW, bottom = -80)
    plt.title('Frequency Domain')
    plt.xlabel('Wavelength lambda (nm)')
    plt.ylabel('Spectrum(dB)')
    plt.xlim([1200, 2400])
    plt.ylim([-80, 20])
    plt.show()
