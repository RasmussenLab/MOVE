import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from matplotlib_venn import venn2, venn3
##################################### Functions #############################################

def plot_confusion_matrix(cm,
                          target_names,
                          cmap=None,
                          normalize=False):
    
    """ Function that plots the confusion matrix given cm. Mattias Ohlsson's code extended."""

    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(4,3))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0, fontsize=12)
        plt.yticks(tick_marks, target_names,fontsize=12)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=14)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=14)


    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontsize=14)

    return fig

def classify_associations(target_file, assoc_tuples):
    self_assoc = 0 # Self associations
    found_assoc_dict = {}
    false_assoc_dict = {}
    tp_fp = np.array([[0,0]])
    with open (target_file, "r") as f:
        for line in f:
            if line[0] != "f":
                splitline = line.strip().split("\t") 
                feat_a = splitline[2]
                feat_b = splitline[3]
                score = abs(float(splitline[5]))
                if feat_a == feat_b: # Self associations will not be counted
                    self_assoc += 1
                else:
                    if (feat_a,feat_b) in assoc_tuples:
                        found_assoc_dict[(feat_a,feat_b)] = score
                        if (feat_b,feat_a) not in found_assoc_dict.keys(): #If we had not found it yet
                            tp_fp = np.vstack((tp_fp,tp_fp[-1]+[0,1]))
                    elif (feat_a,feat_b) not in assoc_tuples:
                        false_assoc_dict[(feat_a,feat_b)] = score
                        if (feat_b,feat_a) not in false_assoc_dict.keys():
                            tp_fp = np.vstack((tp_fp,tp_fp[-1]+[1,0]))

    # Remove duplicated associations:
    for (i,j) in list(found_assoc_dict.keys()):
        if (j,i) in found_assoc_dict.keys():
            del found_assoc_dict[(j,i)] # remove the weakest direction for the association


    for (i,j) in list(false_assoc_dict.keys()):
        if (j,i) in false_assoc_dict.keys():
            del false_assoc_dict[(i,j)]

    return self_assoc, found_assoc_dict, false_assoc_dict, tp_fp


def create_confusion_matrix(n_feat,associations,real_assoc,false_assoc):
    cm = np.empty((2,2))
    # TN: only counting the upper half matrix (non doubled associations)
    cm[0,0] = (n_feat*n_feat-n_feat)/2 - (associations+false_assoc) # Diagonal is discarded
    cm[0,1] = false_assoc
    cm[1,0] = associations- real_assoc
    cm[1,1] = real_assoc

    return cm

def get_precision_recall(found_assoc_dict,false_assoc_dict,associations):
    y_true = []
    y_pred = []

    # True Positives 
    for score in found_assoc_dict.values():
        y_true.append(1)
        y_pred.append(score)
    # False Positives
    for score in false_assoc_dict.values():
        y_true.append(0)
        y_pred.append(score)
    # False negatives
    for _ in range(associations-len(found_assoc_dict)):
        y_true.append(1)
        y_pred.append(0)

    precision, recall, thr = precision_recall_curve(y_true,y_pred) #thr will tell us score values
    avg_prec = average_precision_score(y_true,y_pred)

    return precision, recall, thr, avg_prec

def plot_precision_recall(precision,recall,avg_prec,label, ax):
    ax.scatter(recall,precision, lw=0, marker=".", s=5, edgecolors='none', label = f"{label} - APS:{avg_prec:.2f}")
    ax.legend()
    return ax


def plot_thr_recall(thr, recall,label,  ax):
    ax.scatter(recall[:-1],thr, lw=0, marker=".", s=5, edgecolors='none', label=label)
    ax.legend()
    return ax

def plot_TP_vs_FP(tp_fp, label, ax):
    ax.scatter(tp_fp[:,0],tp_fp[:,1],s=2, label=label, edgecolors='none')
    ax.legend()
    return ax

def plot_filling_order(order_list, last_rank=None):
    
    if last_rank is None:
        last_rank = len(order_list)
    fig = plt.figure()
    order_img = np.zeros((np.max(order_list),len(order_list)))
    for i, element in enumerate(order_list):
        order_img[element-1,i:] = 1

    plt.imshow(order_img[:last_rank,:], cmap="binary")
    plt.xlabel("Correct prediction number")
    plt.ylabel("Association ranking")
    plt.plot(np.arange(last_rank),np.arange(last_rank))
    return fig

def plot_effect_size_matching(assoc_tuples_dict,found_assoc_dict,label, ALGORITHM, ax):


    ground_truth_effects = [assoc_tuples_dict[key] for key in list(found_assoc_dict.keys())]
    predicted_effects = np.array(list(found_assoc_dict.values()))

    if ALGORITHM == 'ttest':
        #Eq 15 on https://doi.org/10.1146/annurev-statistics-031017-100307
        predicted_effects = [-np.log10(p) if p !=0 else -1 for p in predicted_effects]
        predicted_effects[predicted_effects == -1] = np.max(predicted_effects) # Change zeros for max likelihood, -1 as dummy value
        predicted_effects = np.array(predicted_effects)

    max, min  = np.max(predicted_effects), np.min(predicted_effects)
    standarized_pred_effects = (predicted_effects-min)/(max-min)
    ax.scatter(ground_truth_effects,standarized_pred_effects,s=12, edgecolors='none', label=label)
    ax.legend()
    return ax

def plot_venn_diagram(venn, ax, mode = 'all', scale='log'):
    sets = [set(venn[key][mode]) for key in list(venn.keys())]
    labels = (key for key in  list(venn.keys()))

    if len(venn) == 2:
        venn2(sets, labels, ax=ax)
    elif len(venn) == 3:
        venn3(sets, labels, ax=ax)
    else:
        raise ValueError("Unsupported number of input files.")

def plot_upsetplot(venn,assoc_tuples):
    from upsetplot import UpSet
    import pandas as pd
    from matplotlib import cm

    all_assoc = set([association for ALGORITHM in venn.keys() for association in venn[ALGORITHM]['all']])
    columns = ['ground truth']
    columns.extend([ALGORITHM for ALGORITHM in list(venn.keys())])

    df = {}
    for association in all_assoc:
        df[association] = []

        if association in assoc_tuples:
            df[association].append('TP')
        else:
            df[association].append('FP')
        
        for ALGORITHM in list(venn.keys()):
            if association in venn[ALGORITHM]['all']:
                df[association].append(1)
            else:
                df[association].append(0)

    df = pd.DataFrame.from_dict(df, orient='index', columns = columns)
    df = df.set_index([pd.Index(df[ALGORITHM] == 1) for ALGORITHM in list(venn.keys())])
    upset = UpSet(df, intersection_plot_elements=0, show_counts=True)

    upset.add_stacked_bars(by="ground truth", colors=cm.Pastel1,
                       title="Count by ground truth value", elements=10)

    return upset

###################################### Main code ################################################

parser = argparse.ArgumentParser(description='Read two files with ground truth associations and predicted associations.')
parser.add_argument('-p','--perturbed', metavar='pert', type=str, required=True, help='perturbed feature names')
parser.add_argument('-n','--features', metavar='feat', type=int, required=True, help=' total number of features')
parser.add_argument('-r','--reference', metavar='ref', type=str, required=True,  help='path to the ground truth associations file')
parser.add_argument('-t','--targets', metavar='tar', type=str, required=True, nargs='+', help='path to the predicted associations files')
args = parser.parse_args()


# Defining main performance evaluation figures:
fig_0, ax_0 = plt.subplots(figsize=(7,7))
fig_1, ax_1 = plt.subplots(figsize=(7,7))
fig_2, ax_2 = plt.subplots()
fig_3, ax_3 = plt.subplots()

assoc_tuples_dict = {}

# Reading the file with the ground truth changes:
with open (args.reference, "r") as f:
    for line in f:
        if line[0] != "f" and line[0] != "n":
            splitline = line.strip().split("\t") 
            feat_a = splitline[2]
            feat_b = splitline[3]
            assoc_strength = abs(float(splitline[4]))
            # Only can detect associations with perturbed features
            if args.perturbed in feat_a or args.perturbed in feat_b: 
                assoc_tuples_dict[(feat_a,feat_b)] = assoc_strength
                assoc_tuples_dict[(feat_b,feat_a)] = assoc_strength

associations = int(len(assoc_tuples_dict)/2) 
venn = {}
# Count and save found associations
for target_file in args.targets:      

    ALGORITHM = target_file.split('/')[-1].split('_')[3][:-4] 
    self_assoc, found_assoc_dict, false_assoc_dict, tp_fp = classify_associations(target_file,list(assoc_tuples_dict.keys()))
    real_assoc = len(found_assoc_dict) # True predicted associations
    false_assoc = len(false_assoc_dict) # False predicted associations
    total_assoc = real_assoc + false_assoc

    venn[ALGORITHM] = {}
    venn[ALGORITHM]['correct'] = list(found_assoc_dict.keys()) 
    venn[ALGORITHM]['all'] = list(found_assoc_dict.keys()) +  list(false_assoc_dict.keys())

    # Assess ranking of associations (they are doubled in assoc_tuples):
    order_list = [list(assoc_tuples_dict.keys()).index((feat_a,feat_b))//2 for (feat_a,feat_b) in list(found_assoc_dict.keys())]
    fig = plot_filling_order(order_list)
    fig.savefig(f"Order_image_{ALGORITHM}.png", dpi=200)

    ax_0 = plot_effect_size_matching(assoc_tuples_dict, found_assoc_dict, ALGORITHM, ALGORITHM, ax_0)

    # Plot confusion matrix:
    cm = create_confusion_matrix(args.features,associations,real_assoc,false_assoc)
    fig = plot_confusion_matrix(cm,
                          ["No assoc","Association"],
                          cmap=None,
                          normalize=False)

    fig.savefig(f'Confusion_matrix_{ALGORITHM}.png', dpi=100, bbox_inches='tight')

    # Plot precision-recall and TP-FP curves
    precision, recall, thr, avg_prec = get_precision_recall(found_assoc_dict,false_assoc_dict,associations)
    
    ax_1 = plot_precision_recall(precision,recall, avg_prec,ALGORITHM, ax_1)
    ax_2 = plot_TP_vs_FP(tp_fp, ALGORITHM, ax_2)
    ax_3 = plot_thr_recall(thr, recall, ALGORITHM, ax_3)


    # Write results:
    with open('Performance_evaluation_summary_results.txt','a') as f:
        f.write(f" File:  {target_file}\n")
        f.write(f"Ground truth detectable associations (i.e. involving perturbed feature,{args.perturbed}):{associations}\n")
        f.write(f"{total_assoc} unique associations found\n{self_assoc} self-associations were found before filtering\n{real_assoc} were real associations\n{false_assoc} were either false or below the significance threshold\n")
        #print("Correct associations:\n", found_assoc_tuples, "\n")
        f.write(f"Sensitivity:{real_assoc}/{associations} = {real_assoc/associations}\n")
        f.write(f"Precision:{real_assoc}/{total_assoc} = {(real_assoc)/total_assoc}\n")
        f.write(f"Order list:{order_list}\n\n")
        f.write("______________________________________________________\n")


# Edit figures: layout
ax_0.set_xlabel("Real effect")
ax_0.set_ylabel("Predicted effect")
ax_0.set_ylim((-0.02,1.02))
ax_0.set_xlim((0,1.02))
ax_0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=3, fancybox=True, shadow=True)

ax_1.set_xlabel("Recall")
ax_1.set_ylabel("Precision")
ax_1.legend()
ax_1.set_ylim((0,1.05))
ax_1.set_xlim((0,1.05))

ax_2.set_xlabel("False Positives")
ax_2.set_ylabel("True Positives")
ax_2.set_aspect("equal")

ax_3.set_ylabel("Threshold")
ax_3.set_xlabel("Recall")


# Save main figures:
fig_0.savefig("Effect_size_matchin.png", dpi=200)
fig_1.savefig("Precision_recall.png", dpi=200)
fig_2.savefig("TP_vs_FP.png", dpi=200)
fig_3.savefig("thr_vs_recall.png", dpi=200)

# Plotting venn diagram:
if len(venn) == 2 or len(venn) == 3:
    fig_v, ax_v = plt.subplots()
    ax_v = plot_venn_diagram(venn, ax_v, mode = 'correct')
    fig_v.savefig('Venn_diagram.png', dpi=200)

# Plotting UpSet plot
upset = plot_upsetplot(venn, list(assoc_tuples_dict.keys()))
upset.plot()
plt.savefig('UpSet.png', dpi=200)