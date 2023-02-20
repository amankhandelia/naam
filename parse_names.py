import pandas as pd
from histr import Shabdansh
from typing import List, Dict, Tuple


def get_top_n_names(data_path: str = "../naam.csv", n: int = 32000) -> List[str]:
    """This function takes a path to a CSV file of names, and returns the list of the `n` most frequently
    occurring names from that file. It performs some preprocessing on the data before filtering out the invalid names
    and selecting the most frequent names.

    Args:

    - data_path (str): A string representing the path to the CSV file of names. Defaults to "../naam.csv".
    - n (int): An integer representing the number of most frequent names to select. Defaults to 32000.

    Returns:
    - List[str]: A list of strings representing the n most frequent names.
    """
    # read names csv
    names = pd.read_csv(data_path)

    # wrap it with shabdansh for better devanagri string ops
    names["name"] = names["name"].map(Shabdansh)

    # remove names with only one syllable
    names = names[names["name"].map(len) > 1]

    # filter out invalid names
    names["is_valid_name"] = names["name"].apply(lambda name: all(map(Shabdansh.is_valid_cluster, list(name))))
    names = names[names["is_valid_name"]]

    # pick the most frequent names
    names = names.sort_values(by=["count"], ascending=False)["name"][:n].tolist()
    return names


def generate_grapheme_mapping(names: List[Shabdansh], purna_virama="।") -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Generates a mapping of unique graphemes to integers and vice versa for a given list of names.

    Args:
    - names (List[Shabdansh]): A list of names for which the mapping is to be generated.
    - purna_virama (str, optional): The character to be mapped to integer 0. Default is "।".

    Returns:
    - A tuple of two dictionaries:
        - A dictionary (stoi) that maps a grapheme to its integer representation.
        - A dictionary (itos) that maps an integer representation to its corresponding grapheme.

    Example:
    ```
    names = [Shabdansh(_) for _ in ["राम", "अब्दुल", "राजेन्द्र"]]
    stoi, itos = generate_grapheme_mapping(names)
    print(stoi)
    # Output: {'अ': 1, 'जे': 2, 'न्द्र': 3, 'ब्दु': 4, 'म': 5, 'रा': 6, 'ल': 7, '।': 0}
    ```
    """
    # Create a list of lists of graphemes in each name
    grapheme_ls = [list(name) for name in names]

    # Combine all graphemes in the names and remove duplicates
    graphemes = sorted(list(set([grapheme for grapheme_list in grapheme_ls for grapheme in grapheme_list])))

    # Create a dictionary mapping each grapheme to a unique index, starting at 1
    stoi = {grapheme: index + 1 for index, grapheme in enumerate(graphemes)}

    # Set the index of the purna_virama grapheme to 0
    stoi[purna_virama] = 0

    # Create a dictionary mapping each index to its corresponding grapheme
    itos = {index: grapheme for grapheme, index in stoi.items()}

    # Return the grapheme-to-index and index-to-grapheme dictionaries as a tuple
    return stoi, itos