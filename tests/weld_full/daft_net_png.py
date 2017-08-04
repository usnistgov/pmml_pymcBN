from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)

import daft

# Colors.
p_color = {"ec": "#46a546"}
s_color = {"ec": "#f89406"}

pgm = daft.PGM([3.5, 4], origin=[0, .5], grid_unit=2.5)

l = daft.Node("l", "$l$", 1, 4, plot_params=s_color, observed=True)
h = daft.Node("h", "$h$", 2, 4, plot_params=s_color, observed=True)
e = daft.Node("g", "$g$", 3, 4, plot_params=s_color, observed=True)
# l.va = "baseline"
pgm.add_node(l)
pgm.add_node(h)
pgm.add_node(e)
pgm.add_node(daft.Node("sd_l", r"$\sigma_l$", .5, 3.5, plot_params=p_color))
pgm.add_node(daft.Node("mu_l", r"$\mu_l$", 1, 3, plot_params=p_color))
pgm.add_edge("mu_l", "l")
pgm.add_edge("sd_l", "l")

pgm.add_node(daft.Node("sd_h", r"$\sigma_h$", 1.5, 3.5, plot_params=p_color))
pgm.add_node(daft.Node("mu_h", r"$\mu_h$", 2, 3, plot_params=p_color))
pgm.add_edge("mu_h", "h")
pgm.add_edge("sd_h", "h")

pgm.add_node(daft.Node("sd_g", r"$\sigma_g$", 2.5, 3.5, plot_params=p_color))
pgm.add_node(daft.Node("mu_g", r"$\mu_g$", 3, 3, plot_params=p_color))
pgm.add_edge("mu_g", "g")
pgm.add_edge("sd_g", "g")

pgm.add_node(daft.Node("V", "$V$", 2, 2, plot_params=s_color))
pgm.add_node(daft.Node("sd_V", r"$\sigma_V$", 1, 2, plot_params=p_color))
pgm.add_edge("sd_V", "V")
pgm.add_edge("mu_l", "V")
pgm.add_edge("mu_h", "V")
pgm.add_edge("mu_g", "V")


pgm.add_node(daft.Node("E", "$E$", 2, 1, plot_params=s_color))
pgm.add_node(daft.Node("sd_E", r"$\sigma_E$", 1, 1, plot_params=p_color))
pgm.add_node(daft.Node("rho", r"$\rho$", 2.5, 1.5, plot_params=s_color))
pgm.add_edge("sd_E", "E")
pgm.add_edge("rho", "E")
pgm.add_edge("V", "E")

pgm.render()
pgm.figure.savefig("weld_net_img.png", dpi=300)