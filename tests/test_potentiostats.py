import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def test_plot_biologic(df, exp_name):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_title(exp_name)
    ax1.plot(df.index, df.V.values, 'r')
    ax2.plot(df.index, df.I.values, 'b')
    ax1.set_xlabel('datetime [d h]')
    ax1.set_ylabel('$Voltage [V]$', c='r')
    ax2.set_ylabel('Current [A]', c='b')
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %H'))
    
    showme()
    clf()

def test_plot_gamry(df):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    t = np.linspace(0, len(df.index)//(3600/SWEEP_INTERVAL), len(df.index))
    ax1.plot(t, df.V.values, 'r')
    ax2.plot(t, df.I.values, 'b')
    ax1.set_xlabel('t [h]')
    ax1.set_ylabel('$Voltage [V]$', c='r')
    ax2.set_ylabel('Current [A]', c='b')
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)

    showme(dpi=200)
    clf()