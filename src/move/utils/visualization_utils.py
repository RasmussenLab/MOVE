import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import seaborn as sns
import matplotlib

from move.utils import plot_importance
from move.utils.analysis import get_feature_data

def visualize_likelihood(path, nLayers, nHiddens, nDropout, nBeta, nLatents, likelihood_tests):
    """
    Make a plot that visualizes mean likelihoods' on the test set.

    Args:
        path (str):  a string that defines a path to the directory where the results will be saved
        nLayers (list[int]): list of number of layers used in hyperparameter tuning for visualization   
        nHiddens (list): list of number of nodes in hidden layers used in hyperparameter tuning for visualization
        nDropout (list): list of Dropout probablities used in hyperparameter tuning for visualization
        nBeta (list): list of Beta values used in hyperparameter tuning for visualization
        nLatents (list): list of latent space sizes used in hyperparameter tuning for visualization
        likelihood_tests (DefaultDict): A DefaultDict with likelihood results received during hyperarameter tuning for reconstruction
    """    

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
    """
    Make a boxplot that visualizes reconstruction accuracies 

    Args:
        path (str):  a string that defines a path to the directory where the results will be saved
        nLayers (list[int]): list of number of layers used in hyperparameter tuning for visualization   
        nHiddens (list): list of number of nodes in hidden layers used in hyperparameter tuning for visualization
        nDropout (list): list of Dropout probablities used in hyperparameter tuning for visualization
        nBeta (list): list of Beta values used in hyperparameter tuning for visualization
        nLatents (list): list of latent space sizes used in hyperparameter tuning for visualization
        recon_acc (Defaultdict): A DefaultDict with reconstruction accuracy results received during hyperarameter tuning.
        data_type (str): the name of selected data type (train or test) to save results
    """    
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
    
    # Saving a figure
    plt.savefig(path + f"hyperparameters/all_recon_{data_type}.png")

    
    
def draw_boxplot(path, df, title_text, y_label_text, save_fig_name):
    """
    Draw a boxplot to visualize the results of hyperparameter tuning for stability

    Args:
        path (str): a string that defines a path to the directory where the results will be saved
        df (pd.DataFrame): df with hyperparameter tuning for stability results 
        title_text (str): the text to write on the title for the visualization
        y_label_text (str): the text to write on the title for the visualization
        save_fig_name (str): the filename to save the results
    """    
    df = pd.DataFrame(df)
    fig = plt.figure(figsize=(18,14))
    ax = sns.boxplot(data=df, palette = sns.color_palette('colorblind', df.shape[1]))
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45, size=16, horizontalalignment='right')
    plt.title(title_text, size=20)
    plt.ylabel(y_label_text, size=16)
    plt.xlabel('')
    plt.yticks(fontsize=16)
    plt.savefig(path + f"hyperparameters/{save_fig_name}.png", bbox_inches='tight')
    

    

def embedding_plot_discrete(embedding, _type, name, filename, palette=None):
    """
    Visualize 2 dimension representation of latent space of trained model, where data points are colored by the selected categorical feature

    Args:
        embedding (np.array): 2 dimension representation of latent space of trained model
        _type (np.array): data of selected feature
        name (str): feature name
        filename (str): file name to save the results
        palette (seaborn.palettes._ColorPalette, optional): Color palette for a plot. Defaults to None.
    """

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

    plt.savefig(filename)

    
    
def embedding_plot_float(embedding, type, name, file):
    """
    Visualize 2 dimension representation of latent space of trained model, where data points are colored by the selected continuous feature

    Args:
        embedding (np.array): 2 dimension representation of latent space of trained model
        type (np.array): data of selected feature
        name (str): feature name
        filename (str): file name to save the results
    """    
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

    plt.savefig(filename)


def subtitle_legend(ax, legend_format):
    """
    Make a legend for a plot

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): matplotlib object where to add subtytle object
        legend_format (dict): the format of how the legend will be organized 

    Returns:
        matplotlib.legend.Legend: 
    """    
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
    """
    Visualize the training of the model

    Args:
        path (str): a string that defines a path to the directory where the results will be saved
        losses (list): list of losses on train set during the training
        ce (list): list of Binary cross-entropy losses on categorical data of train set during the training
        sse (list): list of sum of squared estimate of errors on continuous data of train set during the training
        KLD (list): list of KLD losses on train set during the training
        epochs (list): list of the range of epochs used in the training
    """    
    # Plot traing error
    fig = plt.figure()
    plt.plot(epochs, losses, '-g', label='loss')
    plt.plot(epochs, ce, '-b', label='CE')
    plt.plot(epochs, sse, '-r', label='SSE')
    plt.plot(epochs, KLD, '-y', label='KLD')
    plt.legend()
    plt.savefig(path + "loss_test.png")
    
def plot_reconstruction_distribs(path, cat_total_recon, all_values, all_names):
    """
    _summary_

    Args:
        path (str): a string that defines a path to the directory where the results will be saved
        cat_total_recon (list[float]): list of floats (from 0 to 1), which corresponds to the fraction of how many samples were correctly reconstructed
        all_values (list[float]): list of floats (from 0 to 1) that corresponds to the cosine similarity between input data and reconstructed data
        all_names (list): list of names of the data classes used
    """    
    # Plot the reconstruction distributions
    df = pd.DataFrame(cat_total_recon + all_values, index=all_names)
    df_t = df.T

    fig = plt.figure(figsize=(25,15))
    ax = sns.boxplot(data=df_t, palette="colorblind", width=0.7)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
    plt.ylabel('Reconstruction accuracy', fontsize=24)
    ax.xaxis.set_tick_params(labelsize=22)
    ax.yaxis.set_tick_params(labelsize=22)
    fig.subplots_adjust(bottom=0.2)
    plt.savefig(path + "reconstruction_accuracy.png")
    plt.close("all")    

# def get_feature_data(data_type, feature_of_interest, cat_list,
#                      con_list, cat_names, con_names):
#     """
#     Get data of selected feature (if the feature is categorical, the categories are converted to be encoded as integers)

#     Args:
#         data_type (str): the feature data type (categorical or continuous) 
#         feature_of_interest (str): the name of feature you want to visualize
#         cat_list (list): list of np.arrays for data of categorical data type
#         con_list (list): list of np.arrays for data of continuous data type
#         cat_names (list): np.array of strings of feature names of categorical data
#         con_names (list): np.array of strings of feature names of continuous data

#     Returns:
#         (tuple): a tuple containing:
#             feature_data (np.array): np.array of feature data
#             headers: list of headers (either cat_names or con_names)
#     Raises:
#         ValueError: Wrong data type was selected (not categorical or continuous)
#     """    

#     if data_type=='categorical':
#         cat_list_integer = [np.argmax(cat, axis=-1) for cat in cat_list]
#         np_data_ints = np.concatenate(cat_list_integer, axis=-1)
#         headers = cat_names
#     elif data_type=='continuous':
#         np_data_ints = np.concatenate(con_list, axis=-1)
#         headers = con_names
#     else:
#         raise ValueError("Wrong data type was selected")
    
#     feature_data = np_data_ints[:,list(headers).index(feature_of_interest)]
    
#     return(feature_data, headers)
    

def visualize_embedding(path, feature_of_interest, embedding, mask, cat_list, 
                        con_list, cat_names, con_names):
    """
    Visualize 2 dimension representation of latent space of trained model

    Args:
        path (str): a string that defines a path to the directory where the results will be saved
        feature_of_interest (str): feature name to visualize
        embedding (np.array): a 2 dimensional representation of latent space of trained model
        mask (np.array): np.array of boaleans, where False values correspond to features that had only NA values.
        cat_list (list): list of np.arrays for data of categorical data type
        con_list (list): list of np.arrays for data of continuous data type
        cat_names (list): np.array of strings of feature names of categorical data
        con_names (list): np.array of strings of feature names of continuous data

    Raises:
        ValueError: feature_of_interest is not in cat_names or con_names
    """    
    if feature_of_interest in cat_names:
        data_type = 'categorical'
    elif feature_of_interest in con_names:
        data_type = 'continuous'
    else:
        raise ValueError("feature_of_interest is not in cat_names or con_names")
    
    
    feature_data = get_feature_data(data_type, feature_of_interest, 
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
    """
    Make a plot for feature importance

    Args:
        path (str):  a string that defines a path to the directory where the results will be saved
        sum_diffs (np.array): for each feature sum of differences of all latent dimensions between existing latent space and new latent space (where the feature is set to NA value)
        features (np.array): np.array of values of data of the features 
        feature_names (np.array): np.array of strings of feature names
        fig_name (str): figure name to save
    """    
    fig = plt.figure(figsize = (20,20))
    plot_importance.summary_plot(sum_diffs, features=features, 
                                 feature_names=feature_names, max_display = 25, 
                                 show = False, size = 30)
    plt.savefig(path + f"results/{fig_name}.png")
    
    
def plot_categorical_importance(path, sum_diffs, cat_list, feature_names, fig_name):
    """
    Make a plot for feature importance of categorical data

    Args:
        path (str): a string that defines a path to the directory where the results will be saved
        sum_diffs (np.array): for each categorical feature sum of differences of all latent dimensions between existing latent space and new latent space (where the feature is set to NA value)
        cat_list (list): list of np.arrays for data of categorical data type
        feature_names (np.array): np.array of strings of feature names
        fig_name (str): figure name to save 
    """    
    
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
                      fig_name=fig_name)
    

    
def plot_continuous_importance(path, train_loader, sum_diffs, feature_names, fig_name):
    """
    Make a plot for feature importance of continuous data

    Args:
        path (str): a string that defines a path to the directory where the results will be saved
        train_loader (Dataloader): Dataloader of training set
        sum_diffs (np.array): for each continuous feature sum of differences of all latent dimensions between existing latent space and new latent space (where the feature is set to NA value) 
        feature_names (np.array): np.array of strings of feature names
        fig_name (str): figure name to save
    """    
    con_all = np.asarray(train_loader.dataset.con_all)
    sum_diffs = np.transpose(sum_diffs)

    f_plot_importance(path=path,
                      sum_diffs=sum_diffs,
                      features=con_all,
                      feature_names=feature_names,
                      fig_name=fig_name)
    

def visualize_indi_var(df_indi_var, version, path):
    """
    Visualizing variation within drugs across all data and specific for each omics

    Args:
        df_indi_var (pd.DataFrame): TODO
        version (str): the subdirectory where the results will be saved
        path (str): the subdirectory where the results will be saved
    """
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



def visualize_drug_similarity_across_all(recon_average_corr_new_all, drug_h, version, path):
    """
    Visualizing the heatmap of similarities within drugs across all data

    Args:
        recon_average_corr_new_all (np.array): TODO
        drug_h (np.array): np.array of strings of feature names data type whose data are changed to test their effects in the pipeline
        version (str): the subdirectory where the results will be saved
        path (str): the subdirectory where the results will be saved
    """    
    cos_sim = cosine_similarity(recon_average_corr_new_all)

    corr = pd.DataFrame(cos_sim, columns = drug_h, index = drug_h)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.set(font_scale=1.5)
    f, ax = plt.subplots(figsize=(10, 10))

    g = sns.clustermap(corr, cmap=cmap, center=0, xticklabels = True,
                      yticklabels = True,
                      linewidths=0, row_cluster=True, col_cluster=True, metric='correlation')

    plt.savefig(path + "results/heatmap_" + version + "_all.pdf", format = 'pdf', dpi = 800)

