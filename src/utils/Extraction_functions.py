import pandas as pd
from Bio import Entrez
from datetime import datetime
import xmltodict


# Extract data from json obtain from pubmed
def data_extractor(json_data: dict) -> pd.DataFrame:
    """
    Extracts data from a JSON object and creates a DataFrame.

    Parameters:
        json_data (dict): A dictionary containing the JSON data.

    Returns:
        pandas.DataFrame: The extracted data in a DataFrame.
    """

    # Basic data path
    data = json_data["PubmedArticleSet"]["PubmedArticle"]["MedlineCitation"]

    # Get article date and convert to datetime
    date = pd.to_datetime(data["DateRevised"]["Year"] +
                          data["DateRevised"]["Month"] + data["DateRevised"]["Day"])

    # Get Article Title
    if isinstance(data["Article"]["ArticleTitle"], dict):
        title = data["Article"]["ArticleTitle"]["#text"]
    else:
        title = data["Article"]["ArticleTitle"]

    # Get Authors
    authors = []
    author_data = data["Article"]["AuthorList"]["Author"]
    if type(author_data) == list:
        for data_dict in author_data:
            name = data_dict["ForeName"] + " " + data_dict["LastName"]
            authors.append(name)
    elif type(author_data) == dict:
        name = author_data["ForeName"] + " " + author_data["LastName"]
        authors.append(name)

    # Get Keywords
    keywords = []
    keywords_path = data["KeywordList"]["Keyword"]
    if isinstance(keywords_path, list):
        for keyword in data["KeywordList"]["Keyword"]:
            words = keyword["#text"]
            keywords.append(words)
    elif isinstance(keywords_path, dict):
        keywords = keywords_path["#text"]

    # Get Abstract
    abstract = data["Article"]["Abstract"]["AbstractText"]
    final_abstract = False
    if isinstance(abstract, list):
        for text in abstract:
            if final_abstract == False:
                final_abstract = text["#text"]
            else:
                final_abstract += f"\n {text['#text']}"
    elif isinstance(abstract, dict):
        final_abstract = abstract["#text"]
    else:
        final_abstract = abstract

    # Get doi or other Locator
    locator_format = False
    locator_number = False
    locators = data["Article"]["ELocationID"]
    if isinstance(locators, list):
        for locator in locators:
            if locator["@EIdType"] == "doi":
                locator_format = locator["@EIdType"]
                locator_number = locator["#text"]
    else:
        locator_format = locators["@EIdType"]
        locator_number = locators["#text"]

    # Create DataFrame
    df = pd.DataFrame([[data["PMID"]["#text"], title, date, data["Article"]["Journal"]["Title"], data["Article"]["Journal"]["ISOAbbreviation"],
                        authors, final_abstract, keywords, locator_format, locator_number]],
                      columns=["PMID", "Title", "Date", "Journal", "Journal_abreviation", "All_authors", "Abstract", "Keywords", "Locator_format", "Locator_number"])
    df["PMID"] = df["PMID"].astype(int)

    return df

# Attack the Pubmed API and get the dataframe
def attack_api_pubmed(email, init_date: ["YYYY/MM/DD"], end_date: ["YYYY/MM/DD", "now"] = "now", max_results: int = 1000, retmax: int = 1000, describe: bool = False):
    """
    Retrieve PubMed article information using the Entrez API and get a DataFrame with util information.

    Parameters:
    - email (str): The email address used for authentication.
    - init_date (List[str]): The initial date range for searching articles in the format ["YYYY/MM/DD"].
    - end_date (List[str], optional): The end date range for searching articles in the format ["YYYY/MM/DD", "now"]. Defaults to "now".
    - max_results (int, optional): The maximum number of results to retrieve. Defaults to 1000.
    - retmax (int, optional): The maximum number of results to retrieve per request. Defaults to 1000.
    - describe (bool, optional): Whether to print progress information during the function execution. Defaults to False.

    Returns:
    - final_df (DataFrame): A DataFrame containing the retrieved article information.
    """

    # Entrez email which is used for authentication
    Entrez.email = email

    # Determinate the date range
    if end_date == "now":
        # The end_date is today
        actual_date = datetime.now().strftime("%Y/%m/%d")
        init_date = pd.to_datetime(init_date).strftime("%Y/%m/%d")
    else:
        actual_date = pd.to_datetime(actual_date).strftime("%Y/%m/%d")
        init_date = pd.to_datetime(init_date).strftime("%Y/%m/%d")

    # List of paper indexs
    all_results = []

    # Search the index in Pubmed
    for retstart in range(0, max_results, retmax):
        handle = Entrez.esearch(
            db="pubmed", term=f'"{init_date}"[Date - Publication] : "{actual_date}"[Date - Publication]', retmax=retmax, retstart=retstart)
        record = Entrez.read(handle)
        handle.close()
        all_results.extend(record["IdList"])

    # Search the paper information
    handle = Entrez.efetch(db="pubmed", id=all_results,
                           retmode="xml", rettype="abstract")
    record = handle.read()
    handle.close()

    # Create a DataFrame
    final_df = pd.DataFrame()

    if describe:
        counter = 1
    # Download the information of each paper
    for id_paper in all_results:
        handle = Entrez.efetch(db="pubmed", id=id_paper)
        record = handle.read()
        handle.close()

        # Convert the XML to a JSON and extract the information
        json_data = xmltodict.parse(record)
        print(f"start paper {counter}/{max_results}")
        try:
            df = data_extractor(json_data)
            final_df = pd.concat([final_df, df])
            if describe:
                print(f"finish paper {counter}/{max_results}")
                counter += 1
        except KeyError:
            if describe:
                print(f"paper {counter}/{max_results} ignored")
                counter += 1
            continue
    return final_df

