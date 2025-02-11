"""
Processing module for amazon kindle reviews dataset dataset
"""
import os
import pandas as pd
import numpy as np

def main():
    # Load the dataset
    file_path = "/mnt/data/your_dataset.csv"  # Replace with actual file path
    df = pd.read_csv(file_path)

    # Create mappings for unique entities
    book_ids = {asin: idx for idx, asin in enumerate(df["asin"].unique())}
    author_ids = {author: idx + len(book_ids) for idx, author in enumerate(df["author"].unique())}
    seller_ids = {seller: idx + len(book_ids) + len(author_ids) for idx, seller in enumerate(df["soldBy"].unique())}
    category_ids = {cat: idx + len(book_ids) + len(author_ids) + len(seller_ids) for idx, cat in
                    enumerate(df["category_name"].unique())}
    year_ids = {year: idx + len(book_ids) + len(author_ids) + len(seller_ids) + len(category_ids) for idx, year in
                enumerate(df["publishedDate"].astype(str).str[:4].unique())}

    # Define relations
    relations = {
        "WrittenBy": 0,
        "SoldBy": 1,
        "BelongsToCategory": 2,
        "PublishedIn": 3,
        "HasRating": 4,
        "HasPrice": 5,
        "IsBestSeller": 6,
        "IsKindleUnlimited": 7,
    }

    # Generate KG triplets
    kg_triplets = []

    for _, row in df.iterrows():
        book_id = book_ids[row["asin"]]

        # Add triplets
        kg_triplets.append((book_id, relations["WrittenBy"], author_ids[row["author"]]))
        kg_triplets.append((book_id, relations["SoldBy"], seller_ids[row["soldBy"]]))
        kg_triplets.append((book_id, relations["BelongsToCategory"], category_ids[row["category_name"]]))
        kg_triplets.append((book_id, relations["PublishedIn"], year_ids[str(row["publishedDate"])[:4]]))

        # Convert rating and price to categorical entities
        rating_entity = int(float(row["stars"]) * 10) + len(book_ids) + len(author_ids) + len(seller_ids) + len(
            category_ids) + len(year_ids)
        price_entity = int(row["price"] * 10) + len(book_ids) + len(author_ids) + len(seller_ids) + len(
            category_ids) + len(
            year_ids) + 50

        kg_triplets.append((book_id, relations["HasRating"], rating_entity))
        kg_triplets.append((book_id, relations["HasPrice"], price_entity))

        # Boolean attributes
        kg_triplets.append((book_id, relations["IsBestSeller"], 1 if row["isBestSeller"] else 0))
        kg_triplets.append((book_id, relations["IsKindleUnlimited"], 1 if row["isKindleUnlimited"] else 0))

    # Convert to DataFrame and save as KG file
    kg_df = pd.DataFrame(kg_triplets, columns=["head", "relation", "tail"])
    kg_file_path = "/mnt/data/kg_final.txt"
    kg_df.to_csv(kg_file_path, sep=" ", header=False, index=False)

    # # Display dataset
    # import ace_tools as tools
    #
    # tools.display_dataframe_to_user(name="Knowledge Graph Triplets", dataframe=kg_df.head(20))
    #
    # kg_file_path

    # Generate train.txt and test.txt for user-item interactions
    # Since your dataset doesn't include user interactions, we will generate synthetic user interactions.

    # Define number of users
    num_users = 5000  # Set an arbitrary number of users

    # Simulate interactions: Each user interacts with 5-10 random books
    np.random.seed(42)
    train_interactions = []
    test_interactions = []

    for user_id in range(num_users):
        interacted_books = np.random.choice(list(book_ids.values()), size=np.random.randint(5, 11), replace=False)

        # Split into train and test (80%-20%)
        split_idx = int(len(interacted_books) * 0.8)
        train_books = interacted_books[:split_idx]
        test_books = interacted_books[split_idx:]

        # Format interactions
        train_interactions.append(f"{user_id} " + " ".join(map(str, train_books)))
        if len(test_books) > 0:
            test_interactions.append(f"{user_id} " + " ".join(map(str, test_books)))

    # Save train.txt
    train_file_path = "/mnt/data/train.txt"
    with open(train_file_path, "w") as f:
        f.write("\n".join(train_interactions))

    # Save test.txt
    test_file_path = "/mnt/data/test.txt"
    with open(test_file_path, "w") as f:
        f.write("\n".join(test_interactions))

    # Create sparsity.split (Optional)
    sparsity_file_path = "/mnt/data/sparsity.split"
    sparsity_groups = {
        "0-5 interactions": [],
        "6-10 interactions": [],
        "11-20 interactions": [],
        "21+ interactions": [],
    }

    for user_id in range(num_users):
        num_interactions = np.random.randint(5, 21)
        if num_interactions <= 5:
            sparsity_groups["0-5 interactions"].append(user_id)
        elif num_interactions <= 10:
            sparsity_groups["6-10 interactions"].append(user_id)
        elif num_interactions <= 20:
            sparsity_groups["11-20 interactions"].append(user_id)
        else:
            sparsity_groups["21+ interactions"].append(user_id)

    with open(sparsity_file_path, "w") as f:
        for key, users in sparsity_groups.items():
            f.write(f"{key}: {len(users)} users\n")
            f.write(" ".join(map(str, users)) + "\n")

    # Display generated files
    file_paths = {
        "train.txt": train_file_path,
        "test.txt": test_file_path,
        "sparsity.split": sparsity_file_path
    }

    print(file_paths)

if __name__ == "__main__":
    main()



