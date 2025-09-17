


    # df_cell_variables = get_timeserie_mean(mcds, filter_alive=True)
    # df_cell_variables_total = get_timeserie_mean(mcds, filter_alive=False)
    # # df_cell_variables has all the relevant data needed for plotting
    # df_cell_variables.to_csv( os.path.join(instance_folder, 'cells_timeseries_alive_mean.csv'), sep=",", header=True, index=False)
    # df_cell_variables_total.to_csv( os.path.join(instance_folder, 'cells_timeseries_total_mean.csv'), sep=",", header=True, index=False)


    # custom_data = doc.getElementsByTagName("time_add_tnf")
    # k1 = round(float(custom_data[0].firstChild.nodeValue), 4)
    # custom_data = doc.getElementsByTagName("duration_add_tnf")
    # k2 = round(float(custom_data[0].firstChild.nodeValue), 4)
    # custom_data = doc.getElementsByTagName("concentration_tnf")
    # k3 = round(float(custom_data[0].firstChild.nodeValue), 4)

   
    # exp_curve, x_experimental, sim_curve, sim_time = curve_comparison(df_time_course, exp_path, instance_folder)

    # plt.figure(figsize=(5, 2.8), dpi=200)
    # # plt.plot(xnew, experimental_curve, df_norm.time, simulation_curve)

    # plt.plot(x_experimental, exp_curve, "-g", label="experimental")
    # plt.plot(sim_time, sim_curve, "-r", label="simulation")
    # plt.legend(loc="upper left")
    # plt.tight_layout()
    # plt.savefig(os.path.join(instance_folder, 'curve_comparison.png'))

   # list_of_variables = ['bound_external_TNFR', 'unbound_external_TNFR', 'bound_internal_TNFR']

    # df_cell_variables[list_of_variables] = df_cell_variables[list_of_variables] / total_receptor

    # plot_molecular_model(df_cell_variables, list_of_variables, axes[1])
    # threshold = 0.5

    # axes[1].hlines(threshold, 0, df_time_course.time.iloc[-1], label="Activation threshold")
    # ax2 = axes[1].twinx()
    # ax2.plot(df_time_tnf.time, df_time_tnf['tnf'], 'r', label="[TNF]")
    # ax2.set_ylabel("[TNF]")
    # ax2.set_ylim([0, max_tnf])
    # axes[1].legend(loc="upper left")
    # ax2.legend(loc="upper right")

    # list_of_variables = ['tnf_node', 'nfkb_node', 'fadd_node']
    # plot_molecular_model(df_cell_variables, list_of_variables, axes[2])
    # axes[2].set_xlabel("time (min)")
    # ax2 = axes[2].twinx()
    # ax2.plot(df_time_tnf.time, df_time_tnf['tnf'], 'r', label="[TNF]")
    # ax2.set_ylabel("[TNF]")
    # ax2.set_ylim([0, max_tnf])
    # axes[2].legend(loc="upper left")
    # ax2.legend(loc="upper right")


    # df_time_course.to_csv(os.path.join(instance_folder, "time_course.tsv"), sep="\t")
    # df_cell_variables.to_csv(os.path.join(instance_folder, "cell_variables.tsv"), sep="\t")
    # df_time_tnf.to_csv(os.path.join(instance_folder, "tnf_time.tsv"), sep="\t")
