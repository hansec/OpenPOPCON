from src import openpopcon


settingsfile = "./resources/examples/MANTA/POPCON_input_example.yaml"
plotsettingsfile = "./resources/plotsettings.yml"
scalinglawfile = "./resources/scalinglaws.yml"

pc = openpopcon.POPCON(settingsfile=settingsfile, plotsettingsfile=plotsettingsfile, scalinglawfile=scalinglawfile)
pc.single_popcon(plot=True)
pc.single_point(-1,0.6,12,True,show=True)