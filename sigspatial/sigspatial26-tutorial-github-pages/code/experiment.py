import os
import openai
from openai import OpenAI
import torch
from test_osm import get_pairs
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from scipy.stats import pearsonr


city_pairs = [
        ("Sydney", "Warrane"),
        ("Melbourne", "Naarm"),
        ("Newcastle", "Mulubinba"),
        ("Perth", "Boorlo"),
        ("Alice Springs", "Mparntwe"),
        ("Bunbury", "Goomburrup"),
        ("Mount Gambier", "Berrin"),
        ("Adelaide", "Tarntanya"),
        ("Brisbane", "Meanjin"),
        ("Albany", "Kinjarling"),
        ("Canberra", "Ngambri"),
        ("Darwin", "Gulumoerrgin"),
        ("Hobart", "Nipaluna")
    ]

embeddings_path = "./datasets-llama/city_embeddings.pkl"

# test question prompt on gpt-4o-mini
def get_gpt_response(city):
    content = "What was the population of " + city + ", Australia in 2021?"
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": content
            }
        ]
    )

    return completion.choices[0].message.content


# test toponyms on chat gpt text-embedding-3-small
def get_city_embedding(city):
    response = client.embeddings.create(
        input=city,
        model="text-embedding-3-small"
    )

    ret = response.data[0].embedding

    return ret

def save_gpt_responses_for_cities(city_pairs, text_file_path, dictionary_file_path):

    # Erase the file's contents before starting
    with open(text_file_path, "w") as file:
        pass  # This just opens and clears the file

    city_responses = {}

    # append data in the loop
    for anglicized, indigenous in city_pairs:
        city_responses[anglicized] = get_gpt_response(anglicized)
        city_responses[indigenous] = get_gpt_response(indigenous)

        with open(text_file_path, "a") as file:
            file.write(anglicized + ": " + city_responses[anglicized] + "\n")
            file.write(indigenous + ": " + city_responses[indigenous] + "\n")
            file.write("\n")

    with open(dictionary_file_path, "wb") as file:
        pickle.dump(city_responses, file)


def get_gpt_responses(file_path):
    with open(file_path, "rb") as file:
        loaded_hashmap = pickle.load(file)

    return loaded_hashmap


def save_embeddings_for_toponyms(city_pairs, embeddings_file_path):
    embeddings = {}

    # append data in the loop
    for anglicized, indigenous in city_pairs:
        embeddings[anglicized] = get_city_embedding(anglicized)
        embeddings[indigenous] = get_city_embedding(indigenous)

    with open(embeddings_file_path, "wb") as file:
        pickle.dump(embeddings, file)

def get_embeddings(file_path):
    with open(file_path, "rb") as file:
        loaded_hashmap = pickle.load(file)

    return loaded_hashmap


def view_embeddings_tsne(embeddings, is_anglicized_or_indigenous, city_pairs):
    """
    Visualizes the embeddings using t-SNE and colors the dots based on whether the city name is Anglicized or Indigenous.

    Parameters:
    - embeddings: dict {city: embedding_vector}
    - is_anglicized_or_indigenous: dict {city: "A" or "I"}
    - city_pairs: list of tuples [(anglicized_city, indigenous_city), ...]

    Returns:
    - A t-SNE scatter plot
    """

    # Collect all city names from the pairs
    all_cities = [city for pair in city_pairs for city in pair]

    # Filter cities with embeddings available
    valid_cities = [city for city in embeddings if any(name in city for name in all_cities)]

    # Extract embeddings and convert to NumPy arrays
    valid_embeddings = np.array([
        embeddings[city].float().cpu().numpy()
        for city in valid_cities
    ])

    # Labels: Anglicized or Indigenous
    labels = []
    for full_city in valid_cities:
        matched_label = None
        for anglicized, indigenous in city_pairs:
            if anglicized in full_city:
                matched_label = "Anglicized"
                break
            elif indigenous in full_city:
                matched_label = "Indigenous"
                break
        if matched_label is None:
            matched_label = "Unknown"
        labels.append(matched_label)

    # Apply t-SNE with dynamic perplexity
    perplexity = min(5, len(valid_embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced_embeddings = tsne.fit_transform(valid_embeddings)

    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7))
    colors = {"Anglicized": "blue", "Indigenous": "green", "Unknown": "gray"}

    for label in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(
            reduced_embeddings[idxs, 0],
            reduced_embeddings[idxs, 1],
            label=label,
            alpha=0.7
        )

    for i, city in enumerate(valid_cities):
        plt.annotate(city, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8)

    plt.title("t-SNE Visualization of City Embeddings")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def view_embeddings_tsne_by_language(embeddings, language_map, city_pairs):
    # Extract city names from city pairs to maintain ordering
    cities = [city for pair in city_pairs for city in pair]  # Flatten list of tuples

    # Get embeddings and labels for only the cities present in city_pairs
    valid_cities = [city for city in cities if city in embeddings]
    valid_embeddings = np.array([embeddings[city] for city in valid_cities])

    # Map cities to their respective language groups
    city_to_language = {city: lang for lang, city_list in language_map.items() for city in city_list}
    valid_languages = [city_to_language[city] for city in valid_cities]

    # Apply t-SNE to reduce dimensions to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)  # Adjust perplexity based on data size
    reduced_embeddings = tsne.fit_transform(valid_embeddings)

    # Assign unique colors to each language
    unique_languages = list(set(valid_languages))
    color_map = {lang: plt.cm.get_cmap("tab20")(i) for i, lang in enumerate(unique_languages)}
    colors = [color_map[lang] for lang in valid_languages]

    # Create plot
    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors, alpha=0.7)

    # Add labels to points
    for i, city in enumerate(valid_cities):
        plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], city, fontsize=9, ha='right')

    # Create legend
    legend_handles = [plt.Line2D([0], [0], marker='o', color=color_map[lang], markersize=8) 
                      for lang in unique_languages]
    plt.legend(legend_handles, unique_languages, title="Language Group")

    # Set title and labels
    plt.title("t-SNE Visualization of Place Names by Indigenous Language Group")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)

    # Show plot
    plt.show()


def compute_silhouette_scores(embeddings, is_anglicized_or_indigenous, city_pairs):
    # Flatten city pairs into a list of all known city names
    all_cities = [city for pair in city_pairs for city in pair]

    # Filter embedding keys that contain any known city name
    valid_cities = [city for city in embeddings if any(name in city for name in all_cities)]
    valid_embeddings = np.array([
        embeddings[city].float().cpu().numpy()
        for city in valid_cities
    ])

    # Assign numerical labels based on whether a known anglicized or indigenous name is in the embedding key
    labels = []
    for full_city in valid_cities:
        label_found = False
        for anglicized, indigenous in city_pairs:
            if anglicized in full_city:
                labels.append(1)  # Anglicized
                label_found = True
                break
            elif indigenous in full_city:
                labels.append(0)  # Indigenous
                label_found = True
                break
        if not label_found:
            labels.append(-1)  # Unknown (not used in silhouette calculation)

    labels = np.array(labels)

    # Filter out unknown labels
    known_indices = labels != -1
    valid_embeddings = valid_embeddings[known_indices]
    labels = labels[known_indices]

    if len(set(labels)) > 1 and len(valid_embeddings) > 1:
        sample_silhouette_values = silhouette_samples(valid_embeddings, labels)

        anglicized_scores = sample_silhouette_values[labels == 1]
        indigenous_scores = sample_silhouette_values[labels == 0]

        mean_anglicized_score = np.mean(anglicized_scores) if len(anglicized_scores) > 0 else None
        mean_indigenous_score = np.mean(indigenous_scores) if len(indigenous_scores) > 0 else None

        print(f"Silhouette Score (Anglicized): {mean_anglicized_score:.4f}" if mean_anglicized_score is not None else "Not enough data for Anglicized")
        print(f"Silhouette Score (Indigenous): {mean_indigenous_score:.4f}" if mean_indigenous_score is not None else "Not enough data for Indigenous")
    else:
        print("Silhouette score cannot be computed (only one cluster present).")



def euclidian_distance(a, b):
    a = torch.tensor(a).float().cpu().numpy()  # convert from list or torch.Tensor
    b = torch.tensor(b).float().cpu().numpy()
    return np.linalg.norm(a - b)

def cosine_sim(a, b):
    a = torch.tensor(a).float().cpu().numpy()
    b = torch.tensor(b).float().cpu().numpy()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def parse_number(x):
        try:
            return int(str(x).replace(",", ""))
        except:
            return np.nan

def get_cosine_similarities():
    embeddings = get_embeddings(embeddings_path)
    hmap = {}

    use_country_format = "country" in embeddings_path.lower()

    for anglicized, indigenous in city_pairs:
        ang_key = f"{anglicized}, Australia" if use_country_format else anglicized
        ind_key = f"{indigenous}, Australia" if use_country_format else indigenous

        if ang_key in embeddings and ind_key in embeddings:
            sim = cosine_sim(embeddings[ang_key], embeddings[ind_key])
            hmap[anglicized] = sim
        else:
            print(f"Missing embedding for: {ang_key} or {ind_key}")

    return hmap


def get_euclidian_embedding_distance():
    embeddings = get_embeddings(embeddings_path)
    hmap = {}

    use_country_format = "country" in embeddings_path.lower()

    for anglicized, indigenous in city_pairs:
        ang_key = f"{anglicized}, Australia" if use_country_format else anglicized
        ind_key = f"{indigenous}, Australia" if use_country_format else indigenous

        if ang_key in embeddings and ind_key in embeddings:
            dist = euclidian_distance(embeddings[ang_key], embeddings[ind_key])
            hmap[anglicized] = dist
        else:
            print(f"Missing embedding for: {ang_key} or {ind_key}")

    return hmap

# correlating embedding distance with ΔMSE (i.e. mse_ind - mse_ang)
def correlate_embeddings_and_error():

    file_path = "./llama_responses.csv"
    df = pd.read_csv(file_path)  

    hmap = get_cosine_similarities()
    dist_hmap = get_euclidian_embedding_distance()

    print(hmap)

    for index, row in df.iterrows():
        ang_resp = parse_number(row["Anglicized_Response"])
        ind_resp = parse_number(row["Indigenous_Response"])
        true_pop = parse_number(row["True_Populations"])
        
        if np.isnan(ind_resp) or np.isnan(ang_resp) or np.isnan(true_pop):
            continue

        mse_ang = (ang_resp - true_pop) ** 2
        mse_ind = (ind_resp - true_pop) ** 2
        delta_mse = mse_ind - mse_ang

        sim = hmap.get(row["Anglicized"], np.nan)
        dist = 1 - sim if sim is not None else np.nan

        euclidian_dist = dist_hmap.get(row["Anglicized"], np.nan)

        df.loc[index, "Anglicized_Response"] = ang_resp
        df.loc[index, "Indigenous_Response"] = ind_resp
        df.loc[index, "True_Populations"] = true_pop
        df.loc[index, "mse_ang"] = mse_ang
        df.loc[index, "mse_ind"] = mse_ind
        df.loc[index, "delta_mse"] = delta_mse
        df.loc[index, "cosine_similarity"] = sim
        df.loc[index, "cosine_distance"] = dist
        df.loc[index, "euclidian_distance"] = euclidian_dist

    # print(df)

    df_clean = df.dropna(subset=["cosine_distance", "delta_mse"])


    # Calculate correlation
    corr, p = pearsonr(df_clean["cosine_distance"], df_clean["delta_mse"])
    print(f"Pearson correlation between cosine distance and ΔMSE: {corr:.3f}, p-value={p:.3f}")

    corr, p = pearsonr(df_clean["euclidian_distance"], df_clean["delta_mse"])
    print(f"Pearson correlation between euclidian embedding distance and ΔMSE: {corr:.3f}, p-value={p:.3f}")

    return df_clean


    # Step 7: Optional plot
    # plt.scatter(df_valid["embedding_distance"], df_valid["delta_mse"])
    # plt.xlabel("Embedding Distance (1 - Cosine Similarity)")
    # plt.ylabel("ΔMSE (Indigenous - Anglicized)")
    # plt.title("Does Embedding Distance Predict Answer Degradation?")
    # plt.grid(True)
    # plt.show()


def correlate_embeddings_and_error_v2():

    file_path = "./llama_responses.csv"
    df = pd.read_csv(file_path)  

    hmap = get_cosine_similarities()
    dist_hmap = get_euclidian_embedding_distance()

    print(dist_hmap)

    for index, row in df.iterrows():
        ang_resp = parse_number(row["Anglicized_Response"])
        ind_resp = parse_number(row["Indigenous_Response"])
        true_pop = parse_number(row["True_Populations"])
        
        if np.isnan(ind_resp) or np.isnan(ang_resp) or np.isnan(true_pop):
            continue

        mse_ang = (ang_resp - true_pop) ** 2
        mse_ind = (ind_resp - true_pop) ** 2

        sim = hmap.get(row["Anglicized"], np.nan)
        dist = 1 - sim if sim is not None else np.nan

        euclidian_dist = dist_hmap.get(row["Anglicized"], np.nan)

        df.loc[index, "Anglicized_Response"] = ang_resp
        df.loc[index, "Indigenous_Response"] = ind_resp
        df.loc[index, "True_Populations"] = true_pop
        df.loc[index, "mse_ang"] = mse_ang
        df.loc[index, "mse_ind"] = mse_ind
        df.loc[index, "cosine_similarity"] = sim
        df.loc[index, "cosine_distance"] = dist
        df.loc[index, "euclidian_distance"] = euclidian_dist

    print(df)

    df_clean = df.dropna(subset=["cosine_distance", "mse_ind", "mse_ang"])

    # Correlation 1: Cosine distance vs Indigenous MSE
    corr_ind, p_ind = pearsonr(df_clean["cosine_distance"], df_clean["mse_ind"])
    print(f"Pearson correlation between cosine distance and Indigenous MSE: {corr_ind:.3f}, p-value={p_ind:.3f}")

    # Correlation 2: Cosine distance vs Anglicized MSE
    corr_ang, p_ang = pearsonr(df_clean["cosine_distance"], df_clean["mse_ang"])
    print(f"Pearson correlation between cosine distance and Anglicized MSE: {corr_ang:.3f}, p-value={p_ang:.3f}")

    # Optional: Euclidian too if useful
    corr_ind_euc, p_ind_euc = pearsonr(df_clean["euclidian_distance"], df_clean["mse_ind"])
    print(f"Pearson correlation between euclidian embedding distance and Indigenous MSE: {corr_ind_euc:.3f}, p-value={p_ind_euc:.3f}")

    corr_ang_euc, p_ang_euc = pearsonr(df_clean["euclidian_distance"], df_clean["mse_ang"])
    print(f"Pearson correlation between euclidian embedding distance and Anglicized MSE: {corr_ang_euc:.3f}, p-value={p_ang_euc:.3f}")

    return df_clean

def correlate_error_diff_with_embedding_distance():
    file_path = "./llama_responses.csv"
    df = pd.read_csv(file_path)

    hmap = get_cosine_similarities()  # Dict: {city_name: cosine_similarity}
    dist_hmap = get_euclidian_embedding_distance()  # Dict: {city_name: euclidean_distance}

    for index, row in df.iterrows():
        ang_resp = parse_number(row["Anglicized_Response"])
        ind_resp = parse_number(row["Indigenous_Response"])
        true_pop = parse_number(row["True_Populations"])
        
        if np.isnan(ind_resp) or np.isnan(ang_resp) or np.isnan(true_pop):
            continue

        mse_ang = (ang_resp - true_pop) ** 2
        mse_ind = (ind_resp - true_pop) ** 2
        error_diff = abs(mse_ang - mse_ind)  # Y: Difference in model error

        cosine_sim = hmap.get(row["Anglicized"], np.nan)
        cosine_dist = 1 - cosine_sim if cosine_sim is not None else np.nan
        euclidian_dist = dist_hmap.get(row["Anglicized"], np.nan)  # X: embedding distance

        df.loc[index, "mse_ang"] = mse_ang
        df.loc[index, "mse_ind"] = mse_ind
        df.loc[index, "error_diff"] = error_diff
        df.loc[index, "cosine_distance"] = cosine_dist
        df.loc[index, "euclidian_distance"] = euclidian_dist

    df_clean = df.dropna(subset=["euclidian_distance", "error_diff"])
    df_clean_cosine = df.dropna(subset=["cosine_distance", "error_diff"])

    print(df_clean)

    # Correlation between Euclidean distance and MSE difference
    corr, p_val = pearsonr(df_clean["euclidian_distance"], df_clean["error_diff"])
    corr_cos, p_val_cos = pearsonr(df_clean_cosine["cosine_distance"], df_clean_cosine["error_diff"])

    print(f"Pearson correlation between euclidian embedding distance and error difference (Anglicized - Indigenous): {corr:.3f}, p-value={p_val:.3f}")
    print(f"Pearson correlation between cosine embedding distance and error difference (Anglicized - Indigenous): {corr_cos:.3f}, p-value={p_val_cos:.3f}")

    return df_clean

if __name__ == "__main__":
    is_anglicized_or_indigenous = {}
    for anglicized, indigenous in city_pairs:
        is_anglicized_or_indigenous[anglicized] = "A"
        is_anglicized_or_indigenous[indigenous] = "I"

    embeddings = get_embeddings(embeddings_path)

    correlate_error_diff_with_embedding_distance()
    compute_silhouette_scores(embeddings, is_anglicized_or_indigenous, city_pairs)
    view_embeddings_tsne(embeddings, is_anglicized_or_indigenous, city_pairs)
