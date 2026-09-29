import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns


with open("H2O_4_4_aug-cc-pvdz_tUPS_L=1_ideal.txt") as f:
    H2O_4_4_aug_cc_pvdz_tUPS_L1_ideal = [line.rstrip() for line in f]

with open("H2O_4_4_aug-cc-pvdz_tUPS_L=2_ideal.txt") as f:
    H2O_4_4_aug_cc_pvdz_tUPS_L2_ideal = [line.rstrip() for line in f]

with open("H2O_4_4_aug-cc-pvdz_classical_methods.txt") as f:
    H2O_4_4_aug_cc_pvdz_classical_methods = [line.rstrip() for line in f]



distance = []
tUPS_L1_ideal_aug_cc_pvdz_4_4 = []
tUPS_L2_ideal_aug_cc_pvdz_4_4 = []

for i in range(len(H2O_4_4_aug_cc_pvdz_tUPS_L1_ideal)):
    distance.append(float(H2O_4_4_aug_cc_pvdz_tUPS_L1_ideal[i].split()[0]) + 0.957848)
    tUPS_L1_ideal_aug_cc_pvdz_4_4.append(float(H2O_4_4_aug_cc_pvdz_tUPS_L1_ideal[i].split()[1]))
    tUPS_L2_ideal_aug_cc_pvdz_4_4.append(float(H2O_4_4_aug_cc_pvdz_tUPS_L2_ideal[i].split()[1]))


distance_cm = []
RHF_aug_cc_pvdz_4_4 = []
OOMP2_aug_cc_pvdz_4_4 = []
OOCCSD_aug_cc_pvdz_4_4 = []
CASSCF_aug_cc_pvdz_4_4 = []


for i in range(int(len(H2O_4_4_aug_cc_pvdz_classical_methods)/4)):
    distance_cm.append(float(H2O_4_4_aug_cc_pvdz_classical_methods[4*i].split()[1]) + 0.957848 )
    RHF_aug_cc_pvdz_4_4.append(float(H2O_4_4_aug_cc_pvdz_classical_methods[4*i].split()[2]))
    OOMP2_aug_cc_pvdz_4_4.append(float(H2O_4_4_aug_cc_pvdz_classical_methods[4*i+1].split()[2]))
    CASSCF_aug_cc_pvdz_4_4.append(float(H2O_4_4_aug_cc_pvdz_classical_methods[4*i+2].split()[2]))
    OOCCSD_aug_cc_pvdz_4_4.append(float(H2O_4_4_aug_cc_pvdz_classical_methods[4*i+3].split()[2]))



# =============================================================================
# Plot styling
# =============================================================================

palette = sns.color_palette("deep")

HF_COLOR = palette[3]         
OOMP2_COLOR = palette[4]       
OOCCSD_COLOR = palette[8]      
CASSCF_COLOR = palette[9]      
EXP_COLOR = "tab:green"


sns.set_theme(
    style="ticks",
    context="paper",
    font_scale=1.8,
    rc={
        "font.family": "Times New Roman",
        "axes.linewidth": 1.8,
        "grid.alpha": 0.15,
    }
)


fig, axs = plt.subplots(
    1, 2,
    figsize=(18, 9),
    sharex=True,
    sharey=True,
)

# =============================================================================
# Common plotting function
# =============================================================================

def plot_panel(ax,
               title,
               distance,
               tups=None,
               tups_noise=None,
               rhf=None,
               oomp2=None,
               casscf=None,
               ooccsd=None):

    # Classical methods
    if rhf is not None:
        ax.plot(distance, rhf,
                color=HF_COLOR,
                lw=3,
                label="HF",
                zorder=1)

    if oomp2 is not None:
        ax.plot(distance, oomp2,
                color=OOMP2_COLOR,
                lw=3,
                label="ooMP2",
                zorder=1)

    if ooccsd is not None:
        ax.plot(distance, ooccsd,
                color=OOCCSD_COLOR,
                lw=3,
                label="ooCCSD",
                zorder=1)

    if casscf is not None:
        ax.plot(distance, casscf,
                color=CASSCF_COLOR,
                lw=3,
                label="CASSCF",
                zorder=1)

    # Quantum methods
    if tups is not None:
        ax.scatter(
            distance,
            tups,
            marker="o",
            s=170,
            facecolors="white",
            edgecolors="black",
            linewidths=2,
            label=r"ootUPS(ideal)",
            zorder=5,
        )

    if tups_noise is not None:
        ax.scatter(
            distance,
            tups_noise,
            marker="X",
            s=170,
            facecolors="white",
            edgecolors="black",
            linewidths=2,
            label=r"ootUPS(shot-noise)",
            zorder=5,
        )


    # Experimental geometry
    ax.axvline(
        0.917,
        color=EXP_COLOR,
        lw=3,
        label="Exp. geometry",
        zorder=0,
    )

    ax.set_title(
        title,
        fontsize=24,
        fontweight="bold",
    )

    ax.grid(True, alpha=0.2)
    sns.despine(ax=ax)


# =============================================================================
# Panels
# =============================================================================


plot_panel(
    ax = axs[0],
    title = "AS(4,4) / aug-cc-pVDZ / L=1",
    distance = distance,
    tups = tUPS_L1_ideal_aug_cc_pvdz_4_4,
    tups_noise = None,
    rhf = RHF_aug_cc_pvdz_4_4,
    oomp2 = OOMP2_aug_cc_pvdz_4_4,
    casscf = CASSCF_aug_cc_pvdz_4_4,
    ooccsd = OOCCSD_aug_cc_pvdz_4_4,
)


plot_panel(
    ax = axs[1],
    title = "AS(4,4) / aug-cc-pVDZ / L=2",
    distance = distance,
    tups = tUPS_L2_ideal_aug_cc_pvdz_4_4,
    tups_noise = None,
    rhf = RHF_aug_cc_pvdz_4_4,
    oomp2 = OOMP2_aug_cc_pvdz_4_4,
    casscf = CASSCF_aug_cc_pvdz_4_4,
    ooccsd = OOCCSD_aug_cc_pvdz_4_4,
)



# =============================================================================
# Labels
# =============================================================================

axs[0].set_ylabel(r"Energy [Hartree]", fontsize=24, fontweight="bold")
axs[1].set_ylabel(r"Energy [Hartree]", fontsize=24, fontweight="bold")

axs[0].set_xlabel(r"O–H distance [$\AA$]", fontsize=24, fontweight="bold")
axs[1].set_xlabel(r"O–H distance [$\AA$]", fontsize=24, fontweight="bold")

for ax in axs.flat:
    ax.tick_params(labelsize=22)

# =============================================================================
# Legend
# =============================================================================

from matplotlib.lines import Line2D

# =============================================================================
# Legends
# =============================================================================

# ----- Classical methods -----
classical_handles = [
    Line2D([0], [0],
           color=HF_COLOR,
           lw=3,
           label="HF"),

    Line2D([0], [0],
           color=OOMP2_COLOR,
           lw=3,
           label="ooMP2"),

    Line2D([0], [0],
           color=OOCCSD_COLOR,
           lw=3,
           label="ooCCSD"),

    Line2D([0], [0],
           color=CASSCF_COLOR,
           lw=3,
           label="CASSCF"),
]

# ----- Quantum methods -----
quantum_handles = [
    Line2D([0], [0],
           marker="o",
           markersize=12,
           markerfacecolor="white",
           markeredgecolor="black",
           markeredgewidth=2,
           linewidth=0,
           label="ootUPS"),
]

# ----- Experimental geometry -----
exp_handle = [
    Line2D([0], [0],
           color=EXP_COLOR,
           lw=3,
           label="Exp. geometry")
]

# ----- Classical legend -----
leg1 = fig.legend(
    handles=classical_handles,
    title="Classical methods",
    loc="upper center",
    bbox_to_anchor=(0.30, 1.00),
    ncol=4,
    fontsize=18,
    title_fontsize=20,
    frameon=True,
    edgecolor="black",
)

# ----- Quantum legend -----
leg2 = fig.legend(
    handles=quantum_handles,
    title="Quantum methods",
    loc="upper center",
    bbox_to_anchor=(0.63, 1.00),
    ncol=1,
    fontsize=18,
    title_fontsize=20,
    frameon=True,
    edgecolor="black",
)

# ----- Experimental geometry legend -----
leg3 = fig.legend(
    handles=exp_handle,
    title="Geometry",
    loc="upper center",
    bbox_to_anchor=(0.85, 1.00),
    ncol=1,
    fontsize=18,
    title_fontsize=20,
    frameon=True,
    edgecolor="black",
)

# =============================================================================
# Layout
# =============================================================================

fig.align_ylabels()

plt.subplots_adjust(
    top=0.75,
    bottom=0.10,
    left=0.08,
    right=0.98,
    wspace=0.18,
    hspace=0.28,
)
plt.show()
