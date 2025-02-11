"""
Script for creatin KG dataset based on amazon kindle
"""
import csv
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD

def main():
    CSV_FILE_PATH = "kindle_books.csv"

    g = Graph()
    KINDLE = Namespace("http://example.org/kindle/")

    g.bind("kindle", KINDLE)

    g.add((KINDLE.Book, RDFS.label, Literal("Book")))
    g.add((KINDLE.Author, RDFS.label, Literal("Author")))
    g.add((KINDLE.publisher, RDFS.label, Literal("publisher")))
    g.add((KINDLE.hasAuthor, RDFS.label, Literal("hasAuthor")))
    g.add((KINDLE.hasCategory, RDFS.label, Literal("hasCategory")))

    with open(CSV_FILE_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f, fieldnames=[
            "asin",
            "title",
            "author",
            "soldBy",
            "imgUrl",
            "productURL",
            "stars",
            "reviews",
            "price",
            "isKindleUnlimited",
            "category_id",
            "isBestSeller",
            "isEditorsPick",
            "isGoodReadsChoice",
            "publishedDate",
            "category_name"
        ])

        for row in reader:
            book_uri = KINDLE[row["asin"]]

            g.add((book_uri, RDF.type, KINDLE.Book))

            g.add((book_uri, KINDLE.asin, Literal(row["asin"], datatype=XSD.string)))
            g.add((book_uri, KINDLE.title, Literal(row["title"], datatype=XSD.string)))
            g.add((book_uri, KINDLE.soldBy, Literal(row["soldBy"], datatype=XSD.string)))
            g.add((book_uri, KINDLE.imgUrl, Literal(row["imgUrl"], datatype=XSD.anyURI)))
            g.add((book_uri, KINDLE.productURL, Literal(row["productURL"], datatype=XSD.anyURI)))
            g.add((book_uri, KINDLE.stars, Literal(row["stars"], datatype=XSD.float)))
            g.add((book_uri, KINDLE.reviews, Literal(row["reviews"], datatype=XSD.integer)))
            g.add((book_uri, KINDLE.price, Literal(row["price"], datatype=XSD.float)))
            g.add((book_uri, KINDLE.isKindleUnlimited, Literal(row["isKindleUnlimited"], datatype=XSD.boolean)))
            g.add((book_uri, KINDLE.category_id, Literal(row["category_id"], datatype=XSD.integer)))
            g.add((book_uri, KINDLE.isBestSeller, Literal(row["isBestSeller"], datatype=XSD.boolean)))
            g.add((book_uri, KINDLE.isEditorsPick, Literal(row["isEditorsPick"], datatype=XSD.boolean)))
            g.add((book_uri, KINDLE.isGoodReadsChoice, Literal(row["isGoodReadsChoice"], datatype=XSD.boolean)))
            g.add((book_uri, KINDLE.publishedDate, Literal(row["publishedDate"], datatype=XSD.date)))
            g.add((book_uri, KINDLE.category_name, Literal(row["category_name"], datatype=XSD.string)))


            author_uri = KINDLE["author/" + row["author"].replace(" ", "_")]
            g.add((author_uri, RDF.type, KINDLE.Author))
            g.add((book_uri, KINDLE.hasAuthor, author_uri))

            category_uri = KINDLE["category/" + row["category_name"].replace(" ", "_")]
            g.add((book_uri, KINDLE.hasCategory, category_uri))

    g.serialize(destination="kindle_books.ttl", format="turtle")
    print("Knowledge graph saved to kindle_books.ttl.")


if __name__ == "__main__":
    main()
