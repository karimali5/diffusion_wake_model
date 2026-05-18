# ============================================================ #
# This script provides an implementation of a diffusion-based  #
# turbine wake model based on (Ali et al. 2024):               #
#                                                              #
#  Ali K, Stallard T, Ouro P. A diffusion-based wind turbine   #
#  wake model. Journal of Fluid Mechanics. 2024;1001:A13.      #
#  doi:10.1017/jfm.2024.1077                                   #
#                                                              #
#  Bibtex entry:                                               #
#   @article{Ali_2024,                                         #
#       title={A diffusion-based wind turbine wake model},     #
#       volume={1001},                                         #
#       DOI={10.1017/jfm.2024.1077},                           #
#       journal={Journal of Fluid Mechanics},                  # 
#       author={Ali, Karim and Stallard, Tim and Ouro, Pablo}, #
#       year={2024},                                           #
#       pages={A13}                                            #
#   }                                                          #
#                                                              #
# Please cite the above article in case this model is used.    #
#                                                              #
# Key points:                                                  #
# ------------                                                 #
# - This model is based on the assumption that normal to the   #
#   streamwise direction, the shape of a turbine's wake        # 
#   behaves similar to the diffusion of a passive scalar.      #
#                                                              #
# - The model naturally evolves from a radially uniform        #
#   shape in the near wake to a Gaussian shape in the far      #
#   wake.                                                      # 
#                                                              #
# - The wake length scale ($\sigma$) is adjusted to take into  #
#   account the near-wake region.                              #
#                                                              #
# - The model presents analytical solutions to integrals of    #
#   the modified Bessel function within the context of         #
#   enforcing the conservation of linear momentum.             #
#                                                              #
# For any enquiries, please contact:                           #
#               karim.ali@manchester.ac.uk                     #
#                                                              #
# December 19, 2024                                            #   
# ============================================================ #

#
# ... Import relevant libraries
#
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt, exp, pi
from scipy.special import erf
# from scipy.special import i0
# import scipy.integrate as integrate

# 
# ... The diffusion-based wake model
# 
def diffusion_model(yds, ct, ti, xd, lnw):
    '''
    This function calculates the normalised wind speed deficit
    in the wake of a turbine following a diffusion-based model.
    
    INPUTS:
    - yds: a list of values for the lateral coordinate y normalised
           by the diameter of the turbine.
    - ct: the thrust coefficient of the turbine.
    - ti: the turbulence intensity of the free-stream flow.
    - xd: the streamwise distance measured from the turbine location,
        normalised by the turbine's diameter.
    - lnw: the streamwise extent of the near wake region, normalised
        by the turbine's diameter.
    
    OUTPUT:
        A list of the same size as "yds" containing the normalised
        wind-speed deficit.  

    NOTE:
        All numbered equations below follow the numbering in (Ali et al. 2024).
    '''

    sqrt2 = sqrt(2)
    
    # Rate of wake expansion "ks" & the parameter "eps" are
    # calculated following:
    #
    #   CATHELAIN, M., BLONDEL, F., JOULIN, P.A. & BOZONNET, P.
    #   2020 Calibration of a super-Gaussian wake model with a
    #   focus on near-wake characteristics. J. Phys.: Conf. Ser.
    #   1618 (6), 062008.
    #
    ks = 0.0119 + 0.18 * ti
    eps = (0.0564 * ct + 0.13) * ((1 + sqrt(1-ct)) / (2 * sqrt(1-ct))) ** (0.5)
    
    # Far wake length scale normalised by the turbine's radius
    # Equation 2.12 in (Ali et al. 2024)
    sigma_fw = 2 * (ks * xd + eps)
    
    # Initial wake length scale 
    # Equation 2.19
    g = eps*(1.0 + 2.0 * exp(-1.0/(8.0 * eps**2)))

    # The value of Lambda at x = 0
    # Following equation 2.10 
    l0 = calc_lam(g, 1)

    # Scaling function C(x) at x = 0
    # Following equation 2.11
    c0 = (1-sqrt(1-ct)) / (1-exp(-1.0/(2.0*g**2)))

    # The size of the actuator disk normalised by the turbine's radius
    # Equation 2.18
    Rd = sqrt(l0 * ct / (1-(1 - l0 * c0)**2))

    # The constant tau
    # Sensitivity to this constant was discussed in Appendix D, 
    # specifically in Fig. 9 (Ali at al. 2024)
    tau = 2

    # The near-wake length scale
    # Equation 2.15
    sigma_nw = Rd * eps * exp(-xd/(tau*lnw)) + sigma_fw * Rd * exp(-0.5/sigma_fw**2)
    
    # Blending the near- and far-wake length scales
    # Equation 2.16
    if xd <= lnw:
        sigma = sigma_nw
    else: 
        wght = exp(-tau*(xd-lnw)/lnw)  
        sigma = wght * sigma_nw + (1-wght) * sigma_fw

    # The parameter Lambda
    # Following equation 2.10 
    lam = calc_lam(sigma, Rd)

    # The streamwise scaling function C(x)
    # Following equation 2.11
    c = (1 - sqrt(1 - lam * ct/Rd**2)) / lam

    # Normalized deficit based om the solution in terms of thr error function. 
    W = []
    for r_ in yds:
        r = r_ * 2 # Normalise by turbine's radius
        gamma = 2*sigma**2 * (1 + 1/(r + 1e-8) + 1/(16 * r**2 + 1e-8)) # The small constant is to avoid division by zero at r = 0
        mu = (sqrt(pi)/2.0)**erf(gamma)
        K = (erf((mu+r)/(sqrt2*sigma)) + erf((mu-r)/(sqrt2*sigma))) / erf(mu/(sqrt2*sigma))
        W.append(c/2 * (1-exp(-Rd**2/(2*sigma**2))) * K)
    
    # A list of normalised deficits for each lateral distance in "yds"
    # Equation 2.2
    # W = [c/sigma**(2)*exp(-0.5*(2*yd)**2/sigma**2)*\
    #     integrate.quad(lambda x: (x*exp(-0.5*x**2/sigma**2)\
    #         *i0(x*(2*yd)/sigma**2)), 0, Rd)[0] for yd in yds]
    
    # Return the list of normalised deficits
    return W

# 
# ... Near-wake region extent
# 
def near_wake_length(ct,ti):
    '''
    A function to calculate the extent of the near-wake region
    based on:

    BASTANKHAH, M. & PORTÉ-AGEL, F. 2016 Experimental and 
    theoretical study of wind turbine wakes in yawed conditions.
    J. Fluid Mech. 806, 506-541.

    INPUTS:
    - ct: the thrust coefficient of the turbine.
    - ti: the turbulence intensity of the free-stream flow.
    
    OUTPUT:
        The streamwise extent of the near-wake region
    '''
    
    # Constants  
    a1 = 0.58
    a2 = 0.077

    # The near-wake length x0
    return (1+sqrt(1-ct))/(sqrt(2)*(4*a1*ti+2*a2*(1-sqrt(1-ct))))

# 
# ... The function Lambda(x)
# 
def calc_lam(sigma,Rd):
    '''
    This function calculates the function Lambda(x), defined in
    equation 2.10 in 
    
    Ali K, Stallard T, Ouro P. A diffusion-based wind turbine
    wake model. Journal of Fluid Mechanics. 2024;1001:A13.
    doi:10.1017/jfm.2024.1077    

    '''
    #return 2*(erf(Rd/sigma)-sigma/(Rd*sqrt(pi))*(1-exp(-Rd**2/sigma**2)))**2

    # Include the error function correction
    mus = (sqrt(pi)/2.0)**erf(2*sigma**2)
    return 2*(erf(mus*Rd/sigma)-sigma/(mus*Rd*sqrt(pi))*(1-exp(-mus**2 * Rd**2/sigma**2)))**2

# 
# ... An example application of the diffusion-based wake model
# 
def example():
    '''
    This function compares the predictions of the diffusion-based
    wake model to the experimental measurements reported in

    SCHREIBER, J., BALBAA, A. & BOTTASSO, C.L. 2020 Brief 
    communication: a double-Gaussian wake model. 
    Wind Energy Sci. 5 (1), 237-244.

    The measurements are for a model G1 turbine operating in 
    a free-stream turbulence intensity of 5% at a thrust coefficient
    of 0.75
    '''

    # Turbine's thrust coefficient
    ct = 0.75

    # Free-stream turbulence intensity
    ti = 0.05

    # Distances downstream of the turbine, normalised by the
    # turbine's diameter
    xds = [1.7,2,3,4,6,9]

    # Lateral distances from the wake center, normalised by the
    # turbine's diameter
    yds = [-2 + i*0.02 for i in range(201)]
    
    # Suppress pop-up of figure. Save an image instead.
    matplotlib.use('Agg')
    fig, axs = plt.subplots(ncols = 3, nrows = 2, figsize = (6.5,5))
    axs = axs.flatten()
    handles = [] # To hold the legend handles
    
    # Loop over streamwise distances
    for icase in range(len(xds)):
        xd = xds[icase]
        ax = axs[icase]

        # Near wake length based on BPA 2016
        ld_BPA16 = near_wake_length(ct, ti)

        # EXP data
        # Returns normalised measured wind speed and the lateral
        # coordinate y (normalised by turbine's diameter)
        defles, yles = exp_data(xd)

        # Plot experimental data
        curve = ax.scatter([1-dd for dd in defles], yles,
            marker="o",s = 16,facecolors='white',
            edgecolor = "tab:red",zorder = 10, label = "Exp.")
        if icase == 0: handles.append(curve)

        # Plot diffusion wake model
        curve, = ax.plot(diffusion_model(yds, ct, ti, xd, ld_BPA16),
            yds, color = "k", linewidth = 1,
            linestyle = "-", label = "present model", zorder = 20)
        if icase == 0: handles.append(curve)

        # Figure formatting
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_xlim([-0.05, 0.65])
        ax.set_ylim([-1.05,1.05])
        ax.set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6])
        ax.set_yticks([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
        if icase in [3,4,5]:
            ax.set_xlabel("$W$")
            ax.set_xticklabels([0,0.1,0.2,0.3,0.4,0.5,0.6])
        else:
            ax.set_xticklabels([])
        if icase in [0,3]:
            ax.set_ylabel("$y/D$")
            ax.set_yticklabels([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
        else:
            ax.set_yticklabels([])
        ax.text(0.3,0.9,"(" + to_letter(icase) + ") $x/D=" + str(xd) + "$",fontsize = 10)
            
    # Save figure
    fig.legend(handles = handles, loc = [0.35, 0.96], 
        ncol=6, prop={'size': 8}, frameon = False)
    fig.subplots_adjust(bottom=0.1, top=0.95, left=0.1,
        right=0.96, wspace=0.1, hspace=0.1)
    fig.savefig("example.png", dpi=300)

# 
# ... Experimental measurements of a G1 model turbine
# 
def exp_data(x):
    '''
    Experimental measurements in the wake of a G1 model turbine as 
    reported in 

    SCHREIBER, J., BALBAA, A. & BOTTASSO, C.L. 2020 Brief 
    communication: a double-Gaussian wake model. 
    Wind Energy Sci. 5 (1), 237-244.

    The turbine operates in a free-stream turbulence intensity of 
    5% at a thrust coefficient of 0.75

    INPUTS:
        the streamwise distance x, normalised by the turbine's diameter

    OUTPUTS:
    - measured normalised wind speed
    - lateral coordinate, normalised by the turbine's diameter
    '''
    if x == 1.7:
        return [0.886399968,0.724214239,0.61609042,0.480935646,
        0.552199073,0.603803623,0.574315308,0.520253399,
        0.488307725,0.468648849,0.483393006,0.56694323,
        0.660322892,0.748787835,0.93063244,0.974864911,
        1.001895866,1.004353226,0.987012987,0.948051948,
        0.584415584,0.5], [-0.642756633,-0.550934559,
        -0.503692767,-0.274636615,-0.184251151,
        -0.093744709,-0.005068055,0.088417462,
        0.177109239,0.270383045,0.36111632,0.451426173,
        0.495099123,0.541131137,0.630836102,0.677140316,
        0.723550387,0.812030452,-0.823336663,-0.690789211,
        -0.457782218,-0.368031968]
    elif x == 2:
        return [0.999114749,0.865675367,0.722785094,
        0.641500804,0.594620408,0.500869086,0.485755237,
        0.544364018,0.583313431,0.568199581,0.508852153,
        0.474078936,0.458965086,0.478255131,0.58110696,
        0.752747638,0.912110681,0.958621754,0.980558616,
        0.992466928,0.93419999,0.676142335],[-0.823844585,
        -0.641023825,-0.551477108,-0.504309776,-0.457354568,
        -0.365777519,-0.274685325,-0.184047684,-0.093288828,
        -0.002196634,0.089168291,0.180381698,0.271473892,
        0.362353962,0.450385506,0.542659535,0.632675956,
        0.676722031,0.720919624,0.814178515,-0.690495971,
        0.494910165]
    elif x == 3:
        return [0.992607792,0.88650274,0.78075696,
        0.73266976,0.670220885,0.581241141,0.521003171,
        0.546990526,0.563388077,0.504826711,0.506862587,
        0.547220829,0.652248064,0.714346879,0.781226778,
        0.879068569,0.914811546,0.98613168,0.935869036,
        0.553420156,0.527738559,0.955598832],[-0.820148681,
        -0.638500665,-0.547676657,-0.501100243,-0.456852649,
        -0.366028641,-0.275204633,-0.184380624,-0.091227796,
        0.181244229,0.272068237,0.362892245,0.453716253,
        0.497963847,0.544540261,0.63536427,0.681940684,
        0.817012286,-0.686153846,-0.003076923,0.089230769,
        0.726153846]
    elif x == 4:
        return [0.973419934,0.886089157,0.827868639,
        0.759944701,0.740537862,0.658058794,0.575579727,
        0.551321178,0.527062629,0.527062629,0.541617758,
        0.561024598,0.609541696,0.682317344,0.740537862,
        0.77935154,0.859404753,0.895792576,0.929754545,
        0.968568224,0.916666667,0.583333333],[-0.818882858,
        -0.636884846,-0.54588584,-0.499219683,-0.454886833,
        -0.363887827,-0.181889815,-0.090890808,0.000108198,
        0.091107204,0.18210621,0.273105216,0.364104223,
        0.455103229,0.499436078,0.546102235,0.637101241,
        0.681434091,0.728100248,0.819099254,-0.679768786,
        -0.27283237]
    elif x == 6:
        return [0.937690743,0.879470225,0.840656546,
        0.813972142,0.772732608,0.70480867,0.665994992,
        0.668420847,0.607774474,0.641736443,0.665994992,
        0.685401831,0.738770639,0.777584318,0.801842867,
        0.860063385,0.87704437,0.903728774,0.947394162,
        0.75,0.641025641,0.615384615],[-0.820148681,
        -0.682748259,-0.638500665,-0.547676657,-0.501100243,
        -0.366028641,-0.275204633,-0.184380624,-0.000403787,
        0.181244229,0.272068237,0.362892245,0.453716253,
        0.497963847,0.544540261,0.63536427,0.681940684,
        0.726188278,0.817012286,-0.452307692,-0.095384615,
        0.089230769]
    elif x == 9:
        return [0.927987323,0.869766805,0.869766805,
        0.850359966,0.830953126,0.801842867,0.772732608,
        0.767880898,0.770306753,0.738770639,0.743622349,
        0.753325769,0.777584318,0.809120432,0.811546287,
        0.826101416,0.850359966,0.864915095,0.884321934,
        0.911006339,0.903846154,0.762820513],[-0.818882858,
        -0.636884846,-0.54588584,-0.50155299,-0.454886833,
        -0.363887827,-0.272888821,-0.181889815,-0.090890808,
        0.000108198,0.091107204,0.18210621,0.364104223,
        0.455103229,0.499436078,0.546102235,0.637101241,
        0.681434091,0.728100248,0.819099254,-0.682851638,
        0.269749518]

def to_letter(i):
    return ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
        "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", 
        "w","x", "y", "z"][i]

example()