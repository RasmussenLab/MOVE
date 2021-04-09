#!/usr/bin/env python

import os, sys
import torch
import numpy as np
import re
import seaborn as sns
#matplotlib.use('agg')
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


### Functions

def plot_error(log_steps, cat_loss, con_loss, cat_dataset_names, con_dataset_names, cols, path, version = 'v1', analysis_type='all'):
   fig = plt.figure()
   for l in range(cat_loss.shape[1]):
      plt.plot(log_steps, cat_loss[:,l], cols[l], label=cat_dataset_names[l])
   
   plt.legend()
   plt.savefig(path + "evaluation/loss_" + version + "_" + analysis_type + "_cat.png")
   plt.clf()
   
   fig = plt.figure()
   for l in range(con_loss.shape[1]):
      plt.plot(log_steps, con_loss[:,l], cols[l], label=con_dataset_names[l])
   
   plt.legend()
   plt.savefig(path + "evaluation/loss_" + version + "_" + analysis_type + "_con.png")
   plt.clf()

def embedding_plot_discrete(embedding, type, name, file, palette=None):
   fig = plt.figure(figsize=(12,8))
   if palette == None:
      palette = sns.color_palette('colorblind', len(np.unique(type)))
   
   ax = sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=type,
                        palette = palette,
                        linewidth=0.1, alpha = 0.8, s=40, edgecolor = 'black')
                        
   
   box = ax.get_position()
   ax.set_position([box.x0, box.y0, box.width * 0.80, box.height]) # resize position
   
   legend_format = {name: np.unique(type)}
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
           handle = label_dict[level]
           new_handles.append(handle)
   
   #Labels are populated from handle.get_label() when we only supply handles as an arg
   legend = ax.legend(handles=new_handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                  fontsize = 14)
   #, title_fontsize = 16
   #Turn off DrawingArea visibility to left justify the text if it contains a subtitle
   for draw_area in legend.findobj(matplotlib.offsetbox.DrawingArea):
       for handle in draw_area.get_children():
           if handle.get_label() in legend_format:
               draw_area.set_visible(False)
   
   return legend

