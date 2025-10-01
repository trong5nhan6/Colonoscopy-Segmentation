import os
import gdown
import zipfile

def get_datasets():
    DATASETS = [
        {
            "name": "bkai-igh-neopolyp",
            "url": "https://drive.google.com/file/d/1Xwpk35Jl-VO26Sy7ii9SCG6N87Oqk9oH/view?usp=drive_link"
        },
        {
            "name": "cvc-clinicdb",
            "url": "https://drive.google.com/file/d/1Bsro4zJRZeycEuPEZ7iv7bSCKM7Nnhtw/view?usp=drive_link"
        },
        {
            "name": "CVC-ColonDB",
            "url": "https://drive.google.com/file/d/1sbdbf5vPFATxn6-gfxt_TsMJ_Ifcqpp7/view?usp=drive_link"
        },
        {
            "name": "ETIS-LaribPolypDB",
            "url": "https://drive.google.com/file/d/114GWMz_OuX14hxteYlREF7LEJklSbwnB/view?usp=drive_link"
        },
        {
            "name": "Kvasir-SEG",
            "url": "https://drive.google.com/file/d/1M90m5MgLINrJHWs89z-xGCHCLp49woOF/view?usp=drive_link"
        },
        {
            "name": "polypgen",
            "url": "https://drive.google.com/file/d/1SqTQpoZCVQuc2NUDzdVl5UtplTX3M84K/view?usp=sharing"
        }
    ]
    return DATASETS


def download_and_extract(name, url, out_dir="./datasets"):
    os.makedirs(out_dir, exist_ok=True)
    output_zip = os.path.join(out_dir, f"{name}.zip")
    extract_path = os.path.join(out_dir, name)

    # Fix: convert Google Drive link to correct format
    if "drive.google.com" in url:
        if "id=" in url:
            file_id = url.split("id=")[1]
        else:
            file_id = url.split("/d/")[1].split("/")[0]
        url = f"https://drive.google.com/uc?id={file_id}"

    print(f"Downloading {name} ...")
    gdown.download(url, output_zip, quiet=False)

    print(f"Extracting {output_zip} ...")
    with zipfile.ZipFile(output_zip, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    os.remove(output_zip)
    print(f"{name} is ready at: {extract_path}")


def choose_and_download(selected=""):
    datasets = get_datasets()
    dataset_names = [d["name"].lower() for d in datasets]

    if selected.strip().lower() == "all":
        targets = dataset_names
    else:
        targets = [x.strip().lower() for x in selected.split(",")]

    for target in targets:
        if target in dataset_names:
            d = next(ds for ds in datasets if ds["name"].lower() == target)
            download_and_extract(d["name"], d["url"])
        else:
            print(f"Dataset '{target}' not found!")