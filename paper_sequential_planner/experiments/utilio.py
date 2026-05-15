import csv


def extract_paths(file_path):
    """
    Reads a TSV file and extracts the path for each source-target pair.
    """
    with open(file_path, "r") as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter="\t")

        paths = []
        for row in reader:
            source = row.get("source")
            target = row.get("target")
            path_str = row.get("path", "")
            if path_str:
                try:
                    nodes = [int(node) for node in path_str.split(",")]
                    print(f"Source: {source}, Target: {target}")
                    print(f"  Path: {nodes}")
                    paths.append(nodes)
                except ValueError as e:
                    print(
                        f"Warning: Could not process path for Source: {source}, Target: {target}. Error: {e}"
                    )
                    paths.append(None)
    return paths


def write_gtsp(paths, output_file):
    pass


def read_gtsp(input_file):
    pass


if __name__ == "__main__":
    tsv_file = "combined_paths.tsv"
    extract_paths(tsv_file)
