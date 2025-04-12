# import requests
# import torch
# import io
# import json
# import base64
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# from matplotlib.lines import Line2D
# import matplotlib.ticker as mticker
# import seaborn as sns
# import os
# # Check the docker run command in case you changed the ports there
# vectorQ_url = 'http://localhost:5000'
# inference_server_url = 'http://localhost:8000/generate'
# gpt_cache_url = 'http://localhost:6050'

# ########################################################################################################################
# ### VectorQ Plotting ###################################################################################################
# ########################################################################################################################
# def get_posteriors(filepath, timestamp, output_folder_path=None):
#     url_posteriors = vectorQ_url + "/get_posteriors"
#     response = requests.get(url_posteriors)
    
#     if response.status_code == 200:
#         data = response.json()
        
#         embedding_posteriors = data.get("embedding_posteriors", [])
        
#         if not embedding_posteriors:
#             print("No posterior data available to plot")
#             return
            
#         if output_folder_path:
#             os.makedirs(output_folder_path, exist_ok=True)
#         else:
#             os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
#         plt.figure(figsize=(16, 6))
        
#         for i, embedding_posterior in enumerate(embedding_posteriors):
#             x_values_embedding = embedding_posterior.get("x_values", [])
#             posterior_embedding = embedding_posterior.get("posterior", [])
#             if x_values_embedding and posterior_embedding:
#                 max_value_embedding = max(posterior_embedding)
#                 if max_value_embedding > 0:
#                     posterior_embedding = [p / max_value_embedding for p in posterior_embedding]
#                 plt.plot(x_values_embedding, posterior_embedding, label=f"Embedding {i}")
        
#         plt.title("Individual Threshold Posteriors")
#         plt.xlabel("Similarity Value")
#         plt.ylabel("Incorrect Cache Hit Likelihood")
        
#         save_path = (output_folder_path if output_folder_path else os.path.dirname(filepath)) + f'/posteriors_individual_{timestamp}.pdf'
#         plt.savefig(save_path, format='pdf')
#         plt.close()
#     else:
#         print(f"Failed to retrieve posteriors. Status Code: {response.status_code}")
#         print(f"Error: {response.text}")

# def get_thresholds(filepath, timestamp, output_folder_path=None):
#     url_posteriors = vectorQ_url + "/get_thresholds"
#     response = requests.get(url_posteriors)

#     if response.status_code == 200:
#         data = response.json()
#         correct_thresholds_all = []
#         incorrect_thresholds_all = []

#         if output_folder_path:
#             thresholds_dir = output_folder_path + "/thresholds_embeddings/"
#             os.makedirs(thresholds_dir, exist_ok=True)
#         else:
#             thresholds_dir = os.path.dirname(filepath) + "/thresholds_embeddings/"
#             os.makedirs(thresholds_dir, exist_ok=True)

#         for idx, row in enumerate(data):
#             correct_thresholds = row.get("correct_thresholds", [])
#             incorrect_thresholds = row.get("incorrect_thresholds", [])

#             if (len(correct_thresholds) + len(incorrect_thresholds) < 3):
#                 continue

#             if correct_thresholds and isinstance(correct_thresholds[0], list):
#                 correct_thresholds = [item for sublist in correct_thresholds for item in sublist]
#             if incorrect_thresholds and isinstance(incorrect_thresholds[0], list):
#                 incorrect_thresholds = [item for sublist in incorrect_thresholds for item in sublist]

#             if 0.0 in incorrect_thresholds:
#                 incorrect_thresholds.remove(0.0)
#             if 1.0 in correct_thresholds:
#                 correct_thresholds.remove(1.0)

#             plt.figure(figsize=(12, 8))
#             plt.rcParams.update({
#                 "font.size": 25,
#                 "axes.titlesize": 25,
#                 "axes.labelsize": 25,
#                 "xtick.labelsize": 25,
#                 "ytick.labelsize": 25,
#                 "legend.fontsize": 24,
#             })
            
#             correct_thresholds_all.extend(correct_thresholds)
#             incorrect_thresholds_all.extend(incorrect_thresholds)

#             if (len(correct_thresholds) < 2) or (len(incorrect_thresholds) < 2) or (len(correct_thresholds) + len(incorrect_thresholds) < 10):
#                 continue
            
#             sns.kdeplot(correct_thresholds, color='green', fill=True, alpha=0.2, bw_adjust=0.25)
#             sns.kdeplot(incorrect_thresholds, color='red', fill=True, alpha=0.2, bw_adjust=0.25)

#             plt.title(f"Embedding {idx}, {len(correct_thresholds)} correct, {len(incorrect_thresholds)} incorrect")
#             plt.xlabel(f"Embedding Similarity Values")
#             plt.ylabel("Probability Density")
#             plt.xlim(0, 1)
#             plt.grid(True, linestyle='--', alpha=0.6)
#             legend_patches = [
#                 Patch(color='green', label='Correctness KDF'),
#                 Patch(color='red', label='Incorrectness KDF'),
#             ]
#             plt.legend(
#                 handles=legend_patches,
#                 loc='upper center',
#                 bbox_to_anchor=(0.5, 1.32), 
#                 ncol=2,
#                 frameon=True,
#                 handletextpad=0.5,
#                 columnspacing=1.0,
#             )
#             plt.tight_layout(rect=[0, 0, 1, 0.99])
            
#             save_path = thresholds_dir + f"embedding_{idx}_{timestamp}.pdf"
#             plt.savefig(save_path, format='pdf')
#             plt.close()
        
#         if (len(correct_thresholds) == 0) or (len(incorrect_thresholds) == 0) or (len(correct_thresholds) + len(incorrect_thresholds) < 10):
#             print("Not enough threshold data points to plot combined graph")
#             return
            
#         plt.figure(figsize=(12, 8))
#         plt.rcParams.update({
#             "font.size": 25,
#             "axes.titlesize": 25,
#             "axes.labelsize": 25,
#             "xtick.labelsize": 25,
#             "ytick.labelsize": 25,
#             "legend.fontsize": 24,
#         })
        
#         sns.kdeplot(correct_thresholds_all, color='green', fill=True, alpha=0.2, bw_adjust=0.25)
#         sns.kdeplot(incorrect_thresholds_all, color='red', fill=True, alpha=0.2, bw_adjust=0.25)

#         plt.title(f"All Embeddings, {len(correct_thresholds_all)} correct, {len(incorrect_thresholds_all)} incorrect")
#         plt.xlabel("Embedding Similarity Values (All Embeddings)")
#         plt.ylabel("Probability Density")
#         plt.xlim(0, 1)
#         plt.grid(True, linestyle='--', alpha=0.6)
#         legend_patches = [
#             Patch(color='green', label='Correctness KDF'),
#             Patch(color='red', label='Incorrectness KDF'),
#         ]
#         plt.legend(
#             handles=legend_patches,
#             loc='upper center',
#             bbox_to_anchor=(0.5, 1.32), 
#             ncol=2,
#             frameon=True,
#             handletextpad=0.5,
#             columnspacing=1.0,
#         )
#         plt.tight_layout(rect=[0, 0, 1, 0.99])
        
#         save_path = (output_folder_path if output_folder_path else os.path.dirname(filepath)) + f"/thresholds_embedding_all_{timestamp}.pdf"
#         plt.savefig(save_path, format='pdf')
#         plt.close()
#     else:
#         print(f"Failed to retrieve thresholds. Status Code: {response.status_code}")
#         print(f"Error: {response.text}")

# def plot_combined_thresholds_and_posteriors(filepath, timestamp, output_folder_path=None):
#     if output_folder_path:
#         combined_dir = output_folder_path + "/combined_plots/"
#         os.makedirs(combined_dir, exist_ok=True)
#     else:
#         combined_dir = os.path.dirname(filepath) + "/combined_plots/"
#         os.makedirs(combined_dir, exist_ok=True)
    
#     url_posteriors = vectorQ_url + "/get_posteriors"
#     posteriors_response = requests.get(url_posteriors)
    
#     url_thresholds = vectorQ_url + "/get_thresholds"
#     thresholds_response = requests.get(url_thresholds)
    
#     if posteriors_response.status_code != 200 or thresholds_response.status_code != 200:
#         print(f"Failed to retrieve data for combined plotting. Posteriors status: {posteriors_response.status_code}, Thresholds status: {thresholds_response.status_code}")
#         return
    
#     posteriors_data = posteriors_response.json()
#     thresholds_data = thresholds_response.json()
    
#     embedding_posteriors = posteriors_data.get("embedding_posteriors", [])
    
#     for idx, embedding_posterior in enumerate(embedding_posteriors):
#         if idx >= len(thresholds_data):
#             continue
            
#         row = thresholds_data[idx]
#         correct_thresholds = row.get("correct_thresholds", [])
#         incorrect_thresholds = row.get("incorrect_thresholds", [])
        
#         if correct_thresholds and isinstance(correct_thresholds[0], list):
#             correct_thresholds = [item for sublist in correct_thresholds for item in sublist]
#         if incorrect_thresholds and isinstance(incorrect_thresholds[0], list):
#             incorrect_thresholds = [item for sublist in incorrect_thresholds for item in sublist]
        
#         if 0.0 in incorrect_thresholds:
#             incorrect_thresholds.remove(0.0)
#         if 1.0 in correct_thresholds:
#             correct_thresholds.remove(1.0)
        
#         if (len(correct_thresholds) < 2) or (len(incorrect_thresholds) < 2) or (len(correct_thresholds) + len(incorrect_thresholds) < 10):
#             continue

#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
#         if correct_thresholds:
#             sns.kdeplot(correct_thresholds, color='green', fill=True, alpha=0.2, bw_adjust=0.25, ax=ax1)
#         if incorrect_thresholds:
#             sns.kdeplot(incorrect_thresholds, color='red', fill=True, alpha=0.2, bw_adjust=0.25, ax=ax1)
        
#         ax1.set_title(f"Embedding {idx}, {len(correct_thresholds)} correct, {len(incorrect_thresholds)} incorrect")
#         ax1.set_ylabel("Density")
#         ax1.grid(True, linestyle='--', alpha=0.6)
#         ax1.legend(handles=[
#             Patch(color='green', label='Correctness KDF'),
#             Patch(color='red', label='Incorrectness KDF'),
#         ])
        
#         x_values = embedding_posterior.get("x_values", [])
#         posterior = embedding_posterior.get("posterior", [])
        
#         if x_values and posterior:
#             max_value = max(posterior)
#             if max_value > 0:
#                 posterior = [p / max_value for p in posterior]
            
#             ax2.plot(x_values, posterior, color='blue', linewidth=2)
#             ax2.fill_between(x_values, 0, posterior, color='blue', alpha=0.2)
#             ax2.set_xlabel("Similarity Value")
#             ax2.set_ylabel("Incorrect Cache Hit Likelihood")
#             ax2.grid(True, linestyle='--', alpha=0.6)
        
#         for ax in [ax1, ax2]:
#             ax.set_xlim(0, 1)
        
#         plt.tight_layout()
        
#         save_path = combined_dir + f"combined_embedding_{idx}_{timestamp}.pdf"
#         plt.savefig(save_path, format='pdf')
#         plt.close(fig)

