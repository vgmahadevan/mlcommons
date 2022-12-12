import os
import json
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages

if not os.path.isdir("images"):
    os.makedirs("images")

project_dir = os.path.join(os.curdir, "project")
images_dir = os.path.join(os.curdir, "images")

for exp in os.listdir(project_dir):
    dir = os.path.join(project_dir, exp)

    try:
        file = open(os.path.join(dir, "mlperf_cloudmask.log"), "r")
    except FileNotFoundError:
        continue

    data = None
    for line in file:
        if "result" in line:
            data = line.rstrip()

    if not data:
        continue
    
    data = json.loads(data[9:])
    
    index = []
    loss = data["value"]["training"]["history"]["loss"]
    accuracy = data["value"]["training"]["history"]["accuracy"]
    val_loss = data["value"]["training"]["history"]["val_loss"]
    val_accuracy = data["value"]["training"]["history"]["val_accuracy"]

    for i in range(len(loss)):
        index.append(i+1)

    # The following creates the plot for the graph for the training loss and accuracy

    fig1, ax1 = plt.subplots()

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Training Loss", color="red")
    ax1.plot(index, loss, color="red")
    ax1.tick_params(axis='y', labelcolor="red")

    ax2 = ax1.twinx()

    ax2.set_ylabel("Training Accuracy", color="blue")
    ax2.plot(index, accuracy, color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")

    plt.title(exp+"\n Training Loss and Accuracy")
    
    plt.savefig(os.path.join(images_dir, exp+"_train.pdf"))
    # same as above but for validation set

    fig2, ax1 = plt.subplots()

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Validation Loss", color="red")
    ax1.plot(index, val_loss, color="red")
    ax1.tick_params(axis='y', labelcolor="red")

    ax2 = ax1.twinx()

    ax2.set_ylabel("Validation Accuracy", color="blue")
    ax2.plot(index, val_accuracy, color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")

    plt.title(exp+"\n Validation Loss and Accuracy")

    plt.savefig(os.path.join(images_dir, exp+"_val.pdf"))

    ''' If you want to save all the images into a single pdf
    pp = PdfPages(os.path.join(os.curdir, "graphs.pdf"))
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
       fig.savefig(pp, format='pdf')
    pp.close()
    '''

    plt.close()

    
