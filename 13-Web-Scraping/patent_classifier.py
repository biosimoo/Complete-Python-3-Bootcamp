import argparse
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def fetch_patent_abstract(patent_number):
    """Fetch the abstract text for a patent from Google Patents."""
    url = f"https://patents.google.com/patent/{patent_number}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Failed to fetch {patent_number}: {exc}")
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    abstract_tag = soup.find("meta", {"name": "DC.description"})
    if abstract_tag and abstract_tag.get("content"):
        return abstract_tag["content"]
    # Fallback: try to get text from abstract section
    abstract_div = soup.find("div", class_="abstract")
    if abstract_div:
        return abstract_div.get_text(strip=True)
    return None


def vectorize_texts(texts):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


def cluster_texts(matrix, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(matrix)
    return labels


def reduce_dimensions(matrix):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(matrix.toarray())
    return coords


def plot_clusters(coords, labels, titles):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    for i, title in enumerate(titles):
        plt.annotate(title, (coords[i, 0], coords[i, 1]))
    plt.title("Patent Clusters")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Patent classifier from Google Patents")
    parser.add_argument("patent_numbers", nargs="*", help="List of patent numbers")
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters")
    args = parser.parse_args()

    if not args.patent_numbers:
        print("Please provide patent numbers as arguments.")
        return

    texts = []
    titles = []
    for num in args.patent_numbers:
        text = fetch_patent_abstract(num)
        if text:
            texts.append(text)
            titles.append(num)

    if len(texts) < 2:
        print("Not enough data fetched to cluster.")
        return

    matrix, _ = vectorize_texts(texts)
    labels = cluster_texts(matrix, n_clusters=args.clusters)
    coords = reduce_dimensions(matrix)
    plot_clusters(coords, labels, titles)


if __name__ == "__main__":
    main()
