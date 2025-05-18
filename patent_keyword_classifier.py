import argparse
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def fetch_patent_text(patent_id: str) -> str:
    """Fetch the patent page and return its text."""
    url = f"https://patents.google.com/patent/{patent_id}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text(separator=" ")


def extract_features(texts):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    return vectorizer.fit_transform(texts)


def cluster_features(matrix, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(matrix)
    return labels


def reduce_to_2d(matrix):
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(matrix.toarray())


def plot_clusters(coords, labels, patent_ids):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis")
    for idx, pid in enumerate(patent_ids):
        plt.annotate(pid, (coords[idx, 0], coords[idx, 1]))
    plt.title("Patent Clusters")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()


def main(patent_ids, n_clusters):
    texts = []
    for pid in patent_ids:
        try:
            texts.append(fetch_patent_text(pid))
        except Exception as exc:
            print(f"Error fetching {pid}: {exc}")
    if not texts:
        print("No patent data retrieved.")
        return
    matrix = extract_features(texts)
    labels = cluster_features(matrix, n_clusters)
    coords = reduce_to_2d(matrix)
    plot_clusters(coords, labels, patent_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify patents by keywords")
    parser.add_argument("patents", nargs="+", help="Patent numbers to fetch")
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters")
    args = parser.parse_args()
    main(args.patents, args.clusters)
