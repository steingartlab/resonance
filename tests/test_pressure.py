import matplotlib.pyplot as plt

def plot_force_resistance(force, resistance):
    plt.scatter(
        force,
        resistance/1000,
        c='k',
        marker='x'
        )
    plt.xlabel('Force [N]')
    plt.ylabel('Resistance [$k\Omega$]')
    plt.title('Resistance')
    showme()
    clf()

def plot_conductance_force(force, conductance, scaling_factor=1E6):
    coef = np.polyfit(conductance, force, 1)
    function = np.poly1d(coef)

    plt.plot(
        conductance*scaling_factor,
        function(conductance), 
        c='k',
        linestyle='dashed'
    )
    plt.scatter(
        conductance*scaling_factor,
        force,
        c='r',
        marker='x'
    )
    plt.ylabel('Force [N]')
    plt.xlabel('Conductance [$\mu S$]')
    plt.title('Conductance')
    showme()
    clf()

def main():
    voltage_parsed = parse_voltage(V_out=voltage_raw)
    resistance = get_resistance(V_out=voltage_parsed)
    plot_force_resistance(
        force=calibration_force,
        resistance=resistance
    )

    conductance = 1 / resistance
    plot_conductance_force(
        force=calibration_force,
        conductance=conductance
    )
    # F = force_from_raw_voltage(voltage_raw=voltage_raw)
    # plt.scatter(conductance*1E6, F)
    # showme()