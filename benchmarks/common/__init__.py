import base64
import io
import json
import os

import matplotlib.pyplot as plt
import requests
import seaborn as sns
import torch
from matplotlib.patches import Patch

# Check the docker run command in case you changed the ports there
vectorQ_url = "http://localhost:5000"
inference_server_url = "http://localhost:8000/generate"
gpt_cache_url = "http://localhost:6050"


########################################################################################################################
### VectorQ Plotting ###################################################################################################
########################################################################################################################
def get_posteriors(filepath, timestamp, output_folder_path=None):
    url_posteriors = vectorQ_url + "/get_posteriors"
    response = requests.get(url_posteriors)

    if response.status_code == 200:
        data = response.json()

        embedding_posteriors = data.get("embedding_posteriors", [])

        if not embedding_posteriors:
            print("No posterior data available to plot")
            return

        if output_folder_path:
            os.makedirs(output_folder_path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        plt.figure(figsize=(16, 6))

        for i, embedding_posterior in enumerate(embedding_posteriors):
            x_values_embedding = embedding_posterior.get("x_values", [])
            posterior_embedding = embedding_posterior.get("posterior", [])
            if x_values_embedding and posterior_embedding:
                max_value_embedding = max(posterior_embedding)
                if max_value_embedding > 0:
                    posterior_embedding = [
                        p / max_value_embedding for p in posterior_embedding
                    ]
                plt.plot(
                    x_values_embedding, posterior_embedding, label=f"Embedding {i}"
                )

        plt.title("Individual Threshold Posteriors")
        plt.xlabel("Similarity Value")
        plt.ylabel("Incorrect Cache Hit Likelihood")

        save_path = (
            output_folder_path if output_folder_path else os.path.dirname(filepath)
        ) + f"/posteriors_individual_{timestamp}.pdf"
        plt.savefig(save_path, format="pdf")
        plt.close()
    else:
        print(f"Failed to retrieve posteriors. Status Code: {response.status_code}")
        print(f"Error: {response.text}")


def get_thresholds(filepath, timestamp, output_folder_path=None):
    url_posteriors = vectorQ_url + "/get_thresholds"
    response = requests.get(url_posteriors)

    if response.status_code == 200:
        data = response.json()
        correct_thresholds_all = []
        incorrect_thresholds_all = []

        if output_folder_path:
            thresholds_dir = output_folder_path + "/thresholds_embeddings/"
            os.makedirs(thresholds_dir, exist_ok=True)
        else:
            thresholds_dir = os.path.dirname(filepath) + "/thresholds_embeddings/"
            os.makedirs(thresholds_dir, exist_ok=True)

        for idx, row in enumerate(data):
            correct_thresholds = row.get("correct_thresholds", [])
            incorrect_thresholds = row.get("incorrect_thresholds", [])

            if len(correct_thresholds) + len(incorrect_thresholds) < 3:
                continue

            if correct_thresholds and isinstance(correct_thresholds[0], list):
                correct_thresholds = [
                    item for sublist in correct_thresholds for item in sublist
                ]
            if incorrect_thresholds and isinstance(incorrect_thresholds[0], list):
                incorrect_thresholds = [
                    item for sublist in incorrect_thresholds for item in sublist
                ]

            if 0.0 in incorrect_thresholds:
                incorrect_thresholds.remove(0.0)
            if 1.0 in correct_thresholds:
                correct_thresholds.remove(1.0)

            plt.figure(figsize=(12, 8))
            plt.rcParams.update(
                {
                    "font.size": 25,
                    "axes.titlesize": 25,
                    "axes.labelsize": 25,
                    "xtick.labelsize": 25,
                    "ytick.labelsize": 25,
                    "legend.fontsize": 24,
                }
            )

            correct_thresholds_all.extend(correct_thresholds)
            incorrect_thresholds_all.extend(incorrect_thresholds)

            if (
                (len(correct_thresholds) < 2)
                or (len(incorrect_thresholds) < 2)
                or (len(correct_thresholds) + len(incorrect_thresholds) < 10)
            ):
                continue

            sns.kdeplot(
                correct_thresholds, color="green", fill=True, alpha=0.2, bw_adjust=0.25
            )
            sns.kdeplot(
                incorrect_thresholds, color="red", fill=True, alpha=0.2, bw_adjust=0.25
            )

            plt.title(
                f"Embedding {idx}, {len(correct_thresholds)} correct, {len(incorrect_thresholds)} incorrect"
            )
            plt.xlabel("Embedding Similarity Values")
            plt.ylabel("Probability Density")
            plt.xlim(0, 1)
            plt.grid(True, linestyle="--", alpha=0.6)
            legend_patches = [
                Patch(color="green", label="Correctness KDF"),
                Patch(color="red", label="Incorrectness KDF"),
            ]
            plt.legend(
                handles=legend_patches,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.32),
                ncol=2,
                frameon=True,
                handletextpad=0.5,
                columnspacing=1.0,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.99])

            save_path = thresholds_dir + f"embedding_{idx}_{timestamp}.pdf"
            plt.savefig(save_path, format="pdf")
            plt.close()

        if (
            (len(correct_thresholds) == 0)
            or (len(incorrect_thresholds) == 0)
            or (len(correct_thresholds) + len(incorrect_thresholds) < 10)
        ):
            print("Not enough threshold data points to plot combined graph")
            return

        plt.figure(figsize=(12, 8))
        plt.rcParams.update(
            {
                "font.size": 25,
                "axes.titlesize": 25,
                "axes.labelsize": 25,
                "xtick.labelsize": 25,
                "ytick.labelsize": 25,
                "legend.fontsize": 24,
            }
        )

        sns.kdeplot(
            correct_thresholds_all, color="green", fill=True, alpha=0.2, bw_adjust=0.25
        )
        sns.kdeplot(
            incorrect_thresholds_all, color="red", fill=True, alpha=0.2, bw_adjust=0.25
        )

        plt.title(
            f"All Embeddings, {len(correct_thresholds_all)} correct, {len(incorrect_thresholds_all)} incorrect"
        )
        plt.xlabel("Embedding Similarity Values (All Embeddings)")
        plt.ylabel("Probability Density")
        plt.xlim(0, 1)
        plt.grid(True, linestyle="--", alpha=0.6)
        legend_patches = [
            Patch(color="green", label="Correctness KDF"),
            Patch(color="red", label="Incorrectness KDF"),
        ]
        plt.legend(
            handles=legend_patches,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.32),
            ncol=2,
            frameon=True,
            handletextpad=0.5,
            columnspacing=1.0,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        save_path = (
            output_folder_path if output_folder_path else os.path.dirname(filepath)
        ) + f"/thresholds_embedding_all_{timestamp}.pdf"
        plt.savefig(save_path, format="pdf")
        plt.close()
    else:
        print(f"Failed to retrieve thresholds. Status Code: {response.status_code}")
        print(f"Error: {response.text}")


def plot_combined_thresholds_and_posteriors(
    filepath, timestamp, output_folder_path=None
):
    if output_folder_path:
        combined_dir = output_folder_path + "/combined_plots/"
        os.makedirs(combined_dir, exist_ok=True)
    else:
        combined_dir = os.path.dirname(filepath) + "/combined_plots/"
        os.makedirs(combined_dir, exist_ok=True)

    url_posteriors = vectorQ_url + "/get_posteriors"
    posteriors_response = requests.get(url_posteriors)

    url_thresholds = vectorQ_url + "/get_thresholds"
    thresholds_response = requests.get(url_thresholds)

    if posteriors_response.status_code != 200 or thresholds_response.status_code != 200:
        print(
            f"Failed to retrieve data for combined plotting. Posteriors status: {posteriors_response.status_code}, Thresholds status: {thresholds_response.status_code}"
        )
        return

    posteriors_data = posteriors_response.json()
    thresholds_data = thresholds_response.json()

    embedding_posteriors = posteriors_data.get("embedding_posteriors", [])

    for idx, embedding_posterior in enumerate(embedding_posteriors):
        if idx >= len(thresholds_data):
            continue

        row = thresholds_data[idx]
        correct_thresholds = row.get("correct_thresholds", [])
        incorrect_thresholds = row.get("incorrect_thresholds", [])

        if correct_thresholds and isinstance(correct_thresholds[0], list):
            correct_thresholds = [
                item for sublist in correct_thresholds for item in sublist
            ]
        if incorrect_thresholds and isinstance(incorrect_thresholds[0], list):
            incorrect_thresholds = [
                item for sublist in incorrect_thresholds for item in sublist
            ]

        if 0.0 in incorrect_thresholds:
            incorrect_thresholds.remove(0.0)
        if 1.0 in correct_thresholds:
            correct_thresholds.remove(1.0)

        if (
            (len(correct_thresholds) < 2)
            or (len(incorrect_thresholds) < 2)
            or (len(correct_thresholds) + len(incorrect_thresholds) < 10)
        ):
            continue

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 12), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )

        if correct_thresholds:
            sns.kdeplot(
                correct_thresholds,
                color="green",
                fill=True,
                alpha=0.2,
                bw_adjust=0.25,
                ax=ax1,
            )
        if incorrect_thresholds:
            sns.kdeplot(
                incorrect_thresholds,
                color="red",
                fill=True,
                alpha=0.2,
                bw_adjust=0.25,
                ax=ax1,
            )

        ax1.set_title(
            f"Embedding {idx}, {len(correct_thresholds)} correct, {len(incorrect_thresholds)} incorrect"
        )
        ax1.set_ylabel("Density")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(
            handles=[
                Patch(color="green", label="Correctness KDF"),
                Patch(color="red", label="Incorrectness KDF"),
            ]
        )

        x_values = embedding_posterior.get("x_values", [])
        posterior = embedding_posterior.get("posterior", [])

        if x_values and posterior:
            max_value = max(posterior)
            if max_value > 0:
                posterior = [p / max_value for p in posterior]

            ax2.plot(x_values, posterior, color="blue", linewidth=2)
            ax2.fill_between(x_values, 0, posterior, color="blue", alpha=0.2)
            ax2.set_xlabel("Similarity Value")
            ax2.set_ylabel("Incorrect Cache Hit Likelihood")
            ax2.grid(True, linestyle="--", alpha=0.6)

        for ax in [ax1, ax2]:
            ax.set_xlim(0, 1)

        plt.tight_layout()

        save_path = combined_dir + f"combined_embedding_{idx}_{timestamp}.pdf"
        plt.savefig(save_path, format="pdf")
        plt.close(fig)


########################################################################################################################
### VectorQ API ########################################################################################################
########################################################################################################################
def send_request_to_vectorQ(data):
    headers = {"Content-Type": "application/json"}
    url_request = vectorQ_url + "/infer"
    response = requests.post(url_request, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}


def clear_client_sessions():
    url_benchmark = vectorQ_url + "/clear_client_sessions"
    response = requests.post(url_benchmark)
    if response.status_code == 200:
        r = response.json()
        return r
    else:
        return {"error": response.text}


def clear_storage():
    url_benchmark = vectorQ_url + "/clear_storage"
    response = requests.post(url_benchmark)
    if response.status_code == 200:
        r = response.json()
        print(r)
        return r
    else:
        return {"error": response.text}


def set_embedding_model(model):
    data = {"embedding_model": model}
    response = requests.post(f"{vectorQ_url}/set_embedding_model", json=data)
    assert response.status_code == 200, "Failed to set embedding model"


def set_generalization_strategy(generalization_strategy):
    data = {"generalization_strategy": generalization_strategy}
    response = requests.post(f"{vectorQ_url}/set_generalization_strategy", json=data)
    assert response.status_code == 200, "Failed to set generalization_strategy"


def set_threshold_strategy(threshold):
    print("FIX THIS - Operation not supported any longer")


def set_threshold(threshold):
    print("FIX THIS - Operation not supported any longer")


def init_vector_store(space, dim, max_elements, ef_construction, m, ef):
    data = {
        "space": space,
        "dim": dim,
        "max_elements": max_elements,
        "ef_construction": ef_construction,
        "m": m,
        "ef": ef,
    }
    response = requests.post(f"{vectorQ_url}/init_vector_store", json=data)
    assert response.status_code == 200, "Failed to init vector store"


def get_config(question):
    url_benchmark = vectorQ_url + "/get_config"
    params = {"question": question}
    response = requests.get(url_benchmark, json=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}


def get_cluster_statistics():
    url_benchmark = vectorQ_url + "/get_cluster_statistics"
    response = requests.get(url_benchmark)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}


def get_row_embeddings(tokenizer, model, row, truncation_limit):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    row_tokens = tokenizer(
        row,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=truncation_limit,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        row_outputs = model(**row_tokens)
        row_embeddings = row_outputs.last_hidden_state.squeeze(0).cpu()

    # Convert to Python list before serialization to avoid numpy dependency issues
    row_embeddings_list = row_embeddings.tolist()

    buffer = io.BytesIO()
    torch.save(row_embeddings_list, buffer)
    buffer.seek(0)

    tensor_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    row_tokens.to("cpu")
    return tensor_base64


def warm_up_inference_server():
    print("Warming Up Inference Server...")
    direct_prompt = {
        "prompt": "Test this server",
        "sentence": "Example entry",
    }
    send_request_to_inference_server(direct_prompt)


def warm_up_vectorq_server():
    print("Warming Up VectorQ Server...")

    # Use a simple Python list instead of a tensor to avoid numpy conversion issues
    simple_embedding = [0.0, 1.0, 2.0]

    buffer = io.BytesIO()
    torch.save(simple_embedding, buffer)
    buffer.seek(0)
    row_embedding_bytes = base64.b64encode(buffer.read()).decode("utf-8")
    print(f"Row Embedding Bytes: {row_embedding_bytes}")

    vectorQ_prompt = {
        "question": "Test this server",
        "rows": ["Example entry"],
        "is_dynamic_threshold": True,
        "output_format": "None",
        "threshold": 1.0,
        "rnd_num_lb": 0.0,
        "rnd_num_ub": 1.0,
        "llm_parameter": {},
        "row_embedding": row_embedding_bytes,
        "candidate_response": "Example response",
    }
    send_request_to_vectorQ(vectorQ_prompt)
    clear_client_sessions()


def parse_line(line):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        print("Error while parsing line")
        return None


########################################################################################################################
### Inference Server ###################################################################################################
### input-format: { 'prompt': '...', 'sentence': '...' }  ##############################################################
########################################################################################################################
def send_request_to_inference_server(data):
    headers = {"Content-Type": "application/json"}
    response = requests.post(inference_server_url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}


########################################################################################################################
### GPT Cache ##########################################################################################################
### input-format: { 'question': '...' }  ###############################################################################
########################################################################################################################
def send_request_to_gpt_cache_server(data):
    headers = {"Content-Type": "application/json"}
    try:
        url = gpt_cache_url + "/infer"
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    except Exception as e:
        return {"error": str(e)}


def clear_gpt_cache():
    headers = {"Content-Type": "application/json"}
    try:
        url = gpt_cache_url + "/reset_cache"
        response = requests.post(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    except Exception as e:
        return {"error": str(e)}
