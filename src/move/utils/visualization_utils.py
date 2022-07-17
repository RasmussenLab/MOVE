import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import seaborn as sns
import matplotlib

from move.utils import plot_importance

def visualize_likelihood(path, nLayers, nHiddens, nDropout, nBeta, nLatents, likelihood_tests):
    # Figure for test error/likelihood
    
    # Define styles for the plot
    ncols = ['navy', 'forestgreen', 'dodgerblue']
    styles = [':', '-', '--']
    batch_size = 10 # here I only tested one batch size due to my small sample size
                    # This can be exhanged with another parameter
    n_rows = len(nLayers)
    n_cols = len(nHiddens)

    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18,15), sharex=True, sharey=True, frameon=False)
    ax3 = fig.add_subplot(111, frameon=False)
    
    # Plotting
    y = 0
    for nl in nLayers:
        x = 0
        for nHidden in nHiddens:
            for drop in nDropout:
                c = ncols[nDropout.index(drop)]
                for b in nBeta:
                    s = styles[nBeta.index(b)]

                    d = []
                    for nLatent in nLatents:
                        combi = str([nHidden] * nl) + "+" + str(nLatent) + ", drop: " + str(drop) + \
                                   ", b: " + str(b) + ", batch: " + str(batch_size)
                        d.append(likelihood_tests[combi][0])

                    a = axes[x,y].plot(nLatents, d, label='Dropout: ' + str(drop) +
                                             ', beta: ' + str(b), linestyle=s, color = c, linewidth=2)
            x += 1
            
        y += 1
     
     
     ### Annotating the graph
    
    # Adding annotations of variables on right side of the subplots on y axis
    for i in range(n_rows):
        axes[i, n_cols-1].annotate(nHiddens[i], xy=(1.02, 0.55), xycoords='axes fraction',
                                fontsize = 14, rotation=-90)
    
    # Adding annotation of variable on right side of the graph
    plt.figtext(0.78, 0.40, 'Number of hidden neurons in each layer', rotation=-90, fontsize=14)
    
    # Adding annotations of variables on top of the subplots on x axis
    for i in range(n_cols):
        axes[0,i].set_title(nLayers[i], fontsize=14)
    
    # Setting the size of variables on ticks
    for i in range(n_rows):
        for j in range(n_cols):
            axes[i,j].xaxis.set_tick_params(labelsize=14)
            axes[i,j].yaxis.set_tick_params(labelsize=14)
    plt.xticks(nLatents, nLatents, fontsize=14)
    ax3.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    
    # Setting labels on X and Y axis
    ax3.set_xlabel('Number of latent neurouns', fontsize = 14)
    ax3.set_ylabel('Log-likelihood', fontsize = 14)
    ax3.set_title('Number of hidden layers', fontsize = 14)
    ax3.title.set_position([.5, 1.03])
    ax3.yaxis.set_label_coords(-0.04, 0.5)
    ax3.xaxis.set_label_coords(0.5, -0.03)
    
    # Adding additional adjustments to the graph
    plt.grid(False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.tight_layout()
    fig.subplots_adjust(right=0.75)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize = 14)
    
    # Saving the figure
    plt.savefig(path + "hyperparameters/all_likelihoods_test1.png")

    
def visualize_recon_acc(path, nLayers, nHiddens, nDropout, nBeta, nLatents, recon_acc, data_type):
    # Plot results for train reconstructions
    n_rows = len(nLayers)
    n_cols = len(nHiddens)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(40,30), sharex=True, sharey=True, frameon=False)
    ax3 = fig.add_subplot(111, frameon=False)

    y = 0
    nBeta = [nBeta[0]] # Beta didn't really effect the results so for the reconstructions we only looked at one value
    batch_size = 10
    # Plotting
    for nl in nLayers:
        x = 0
        for nHidden in nHiddens:
            frame = pd.DataFrame()
            indexes = []
            for drop in nDropout:
                for b in nBeta:
                    for nLatent in nLatents:
                        combi = str([nHidden] * nl) + "+" + str(nLatent) + ", drop: " + str(drop) + ", b: " + str(b) + ", batch: " + str(batch_size)
                        r = recon_acc[combi][0]
                    
                        name = 'Latent: ' + str(nLatent) + ', Dropout: ' + str(drop)
                        indexes.append(name)
                        frame[combi] = r

            frame.set_axis(indexes, axis=1, inplace=True)
            sns.boxplot(data=frame, palette="colorblind", width=0.7, ax = axes[x,y])    
          
            x += 1
        y += 1

    ### Anotating
    # Adding annotations of variables on right side of the subplots on y axis
    for i in range(n_rows):
        axes[i,n_cols-1].annotate(nHidden, xy=(1.02, 0.55), xycoords='axes fraction', fontsize = 24, rotation=-90)

    # Adding annotation of variable on right side of the graph
    plt.figtext(0.92, 0.45, 'Number of hidden neurons in each layer', rotation=-90, fontsize=24)
    
    # Adding annotations of variables on top of the subplots on x axis
    for i in range(n_cols):
        axes[0,i].set_title(nLayers[i], fontsize=24)
    
    # Setting the size of variables on ticks
    for i in range(n_rows):
        for j in range(n_cols):
            axes[i,j].set_xticklabels(indexes,fontsize=24, rotation=40, ha="right")
            axes[i,j].yaxis.set_tick_params(labelsize=24)
    
    # Setting labels on X and Y axis
    ax3.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax3.set_ylabel('Reconstruction accuracy', fontsize = 24)
    ax3.set_title('Number of hidden layers', fontsize = 24)
    
    # Adding additional adjustments to the graph
    ax3.title.set_position([.5, 1.03])
    ax3.yaxis.set_label_coords(-0.04, 0.5)
    ax3.xaxis.set_label_coords(0.5, -0.03)
    plt.grid(False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.show()
    
    # Saving a figure
    plt.savefig(path + f"hyperparameters/all_recon_{data_type}.png")

    
def plot_graphs():
    fig = plt.figure()
    plt.plot(epochs, loss, '-g', label='loss')
    plt.plot(epochs, ce, '-b', label='CE')
    plt.plot(epochs, sse, '-r', label='SSE')
    plt.plot(epochs, KLD, '-y', label='KLD')
    plt.legend()
    plt.savefig(path + "/evaluation/loss_" + combi +".png")
    plt.clf()

    fig = plt.figure()
    plt.plot(epochs, loss, '-b', label='loss')
    plt.plot(batchsteps, loss_test, '-r', label='test loss')
    plt.legend()
    plt.savefig(path + "/evaluation/test_loss_" + combi +".png")
    plt.clf()    
    
    
def draw_boxplot(path, df, title_text, y_label_text, save_fig_name):
    
    df = pd.DataFrame(df)
    fig = plt.figure(figsize=(18,14))
    ax = sns.boxplot(data=df, palette = sns.color_palette('colorblind', df.shape[1]))
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45, size=16, horizontalalignment='right')
    plt.title(title_text, size=20)
    plt.ylabel(y_label_text, size=16)
    plt.xlabel('')
    plt.yticks(fontsize=16)
    plt.savefig(path + f"hyperparameters/{save_fig_name}.png")
    

    

def embedding_plot_discrete(embedding, _type, name, file, palette=None):
    fig = plt.figure(figsize=(12,8))
    if palette == None:
        palette = sns.color_palette('colorblind', len(np.unique(_type)))

    ax = sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=_type,
                         palette = palette,
                         linewidth=0.1, alpha = 0.8, s=40, edgecolor = 'black')


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.80, box.height]) # resize position

    legend_format = {name: np.unique(_type)}
    leg = subtitle_legend(ax, legend_format=legend_format)

    plt.xlabel('Dim1')
    plt.ylabel('Dim2')
    plt.style.use('default')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    plt.savefig(file)

    
    
def embedding_plot_float(embedding, type, name, file):
    fig, ax = plt.subplots(figsize=(12,8))
    points = ax.scatter(x=embedding[:,0], y=embedding[:,1], c=type, s=40, cmap="Spectral_r",
                       edgecolor = 'black', linewidth=0.1)

    cbar = fig.colorbar(points, fraction=0.03, pad=0.03)
    cbar.ax.set_title(name, rotation=0, fontsize = 16)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.90, box.height])

    plt.xlabel('Dim1')
    plt.ylabel('Dim2')
    plt.style.use('default')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    plt.savefig(file)


def subtitle_legend(ax, legend_format):
    new_handles = []
   
    handles, labels = ax.get_legend_handles_labels()
    label_dict = dict(zip(labels, handles))
   
   #Means 2 labels were the same
    if len(label_dict) != len(labels):
        raise ValueError("Can not have repeated levels in labels!")
     
    for subtitle, level_order in legend_format.items():
       #Roll a blank handle to add in the subtitle
        blank_handle = matplotlib.patches.Patch(visible=False, label=subtitle)
        new_handles.append(blank_handle)
       
        for level in level_order:
            handle = label_dict[str(level)]
            new_handles.append(handle)
   
   #Labels are populated from handle.get_label() when we only supply handles as an arg
    legend = ax.legend(handles=new_handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                   fontsize = 14)
   #Turn off DrawingArea visibility to left justify the text if it contains a subtitle
    for draw_area in legend.findobj(matplotlib.offsetbox.DrawingArea):
        for handle in draw_area.get_children():
            if handle.get_label() in legend_format:
                draw_area.set_visible(False)
   
    return legend

def visualize_training(path, losses, ce, sse, KLD, epochs):
    # Plot traing error
    fig = plt.figure()
    plt.plot(epochs, losses, '-g', label='loss')
    plt.plot(epochs, ce, '-b', label='CE')
    plt.plot(epochs, sse, '-r', label='SSE')
    plt.plot(epochs, KLD, '-y', label='KLD')
    plt.legend()
    plt.show()
    plt.savefig(path + "loss_test.png")
    
def plot_reconstruction_distribs(path, cat_total_recon, all_values):
    
    # Plot the reconstruction distributions
    df = pd.DataFrame(cat_total_recon + all_values, index = ['Clinical\n(categorical)', 'Genomics', 'Drug data', 'Clinical\n(continuous)', 'Diet +\n wearables','Proteomics','Targeted\nmetabolomics','Untargeted\nmetabolomics', 'Transcriptomics', 'Metagenomics'])
    df_t = df.T

    fig = plt.figure(figsize=(25,15))
    ax = sns.boxplot(data=df_t, palette="colorblind", width=0.7)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
    plt.ylabel('Reconstruction accuracy', fontsize=24)
    ax.xaxis.set_tick_params(labelsize=22)
    ax.yaxis.set_tick_params(labelsize=22)
    fig.subplots_adjust(bottom=0.2)
    plt.savefig(path + "reconstruction_accuracy.png")
    plt.show()
    plt.close("all")    
    
def get_feature_data(data_type, feature_of_interest, cat_list,
                     con_list, cat_names, con_names):
    
    if data_type=='categorical':
        cat_list_integer = [np.argmax(cat, axis=-1) for cat in cat_list]
        np_data_ints = np.concatenate(cat_list_integer, axis=-1)
        headers = cat_names
    elif data_type=='continuous':
        np_data_ints = np.concatenate(con_list, axis=-1)
        headers = con_names
    else:
        raise ValueError("Wrong data type was selected")
    
    feature_data = np_data_ints[:,list(headers).index(feature_of_interest)]
    
    return(feature_data, headers)
    

def visualize_embedding(path, feature_of_interest, embedding, mask, cat_list, 
                        con_list, cat_names, con_names):
    
    if feature_of_interest in cat_names:
        data_type = 'categorical'
    elif feature_of_interest in con_names:
        data_type = 'continuous'
    else:
        raise ValueError("feature_of_interest is not in cat_names or con_names")
    
    
    feature_data, headers = get_feature_data(data_type, feature_of_interest, 
                                             cat_list=cat_list, con_list=con_list, 
                                             cat_names=cat_names, con_names=con_names)
    
    
    if data_type == 'categorical':
        embedding_plot_discrete(embedding, feature_data[mask], 
                                f"{feature_of_interest} (yes/no)", 
                                path + f"results/umap_{feature_of_interest}.png")
        
    elif data_type =='continuous':
        embedding_plot_float(embedding, feature_data[mask], feature_of_interest,
                             path + f"results/umap_{feature_of_interest}.png")
        
        
def f_plot_importance(path, sum_diffs, features, feature_names, fig_name):
    fig = plt.figure(figsize = (20,20))
    plot_importance.summary_plot(sum_diffs, features=features, 
                                 feature_names=feature_names, max_display = 25, 
                                 show = False, size = 30)
    plt.savefig(path + f"results/{fig_name}.png")
    
    
def plot_categorical_importance(path, sum_diffs, cat_list, feature_names, fig_name):
    
    # Converting from one hot to numerical variables
    cat_ints_list = []
    for cat in cat_list:
        cat_target = np.argmax(cat, 2)
        cat_target[np.sum(cat, 2) == 0] = -1
        cat_ints_list.append(cat_target)
    
    cat_target_all = np.concatenate(cat_ints_list, axis=1)
    sum_diffs = np.transpose(sum_diffs)
    
    f_plot_importance(path=path,
                      sum_diffs=sum_diffs,
                      features=cat_target_all,
                      feature_names=feature_names,
                      fig_name='importance_SHAP_cat')
    

    
def plot_continuous_importance(path, train_loader, sum_diffs, feature_names, fig_name):
    
    con_all = np.asarray(train_loader.dataset.con_all)
    sum_diffs = np.transpose(sum_diffs)

    f_plot_importance(path=path,
                      sum_diffs=sum_diffs,
                      features=con_all,
                      feature_names=feature_names,
                      fig_name='importance_SHAP_con')
    

def visualize_indi_var(df_indi_var, version, path):

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(12,10))
    ax = sns.barplot(data=df_indi_var.T, palette="tab10", saturation=0.50)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=60, 
                      ha="right", rotation_mode="anchor")
    plt.ylabel('Patient variance')
    fig.subplots_adjust(bottom=0.2)
    fig.tight_layout()
    plt.grid(False)
    plt.savefig(path + "results/drug_individual_variations_" + version + ".pdf", format = 'pdf', dpi = 800)

    plt.show()   


def visualize_drug_similarity_across_all(recon_average_corr_new_all, drug_h, version, path):
    
    cos_sim = cosine_similarity(recon_average_corr_new_all)

    corr = pd.DataFrame(cos_sim, columns = drug_h, index = drug_h)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.set(font_scale=1.5)
    f, ax = plt.subplots(figsize=(10, 10))

    g = sns.clustermap(corr, cmap=cmap, center=0, xticklabels = True,
                      yticklabels = True,
                      linewidths=0, row_cluster=True, col_cluster=True, metric='correlation')

    plt.savefig(path + "results/heatmap_" + version + "_all.pdf", format = 'pdf', dpi = 800)

