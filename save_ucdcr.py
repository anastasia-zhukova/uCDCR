import os
import shutil
from tqdm import tqdm
from setup import *
from create_summary import check_datasets
from huggingface_hub import login, upload_folder

# login()

DIRECTORIES_TO_OUTPUT = {
       NEWSWCL50: os.path.join(os.getcwd(), NEWSWCL50, OUTPUT_FOLDER_NAME),
       ECB_PLUS: os.path.join(os.getcwd(), ECB_PLUS, OUTPUT_FOLDER_NAME),
       MEANTIME: os.path.join(os.getcwd(), MEANTIME, OUTPUT_FOLDER_NAME),
       NP4E: os.path.join(os.getcwd(), NP4E, OUTPUT_FOLDER_NAME),
       NIDENT: os.path.join(os.getcwd(), NIDENT, OUTPUT_FOLDER_NAME),
       GVC: os.path.join(os.getcwd(), GVC, OUTPUT_FOLDER_NAME),
       FCC: os.path.join(os.getcwd(), FCC, OUTPUT_FOLDER_NAME),
       CD2CR: os.path.join(os.getcwd(), CD2CR, OUTPUT_FOLDER_NAME),
       WEC_ENG: os.path.join(os.getcwd(), WEC_ENG, OUTPUT_FOLDER_NAME),
       CEREC: os.path.join(os.getcwd(), CEREC, OUTPUT_FOLDER_NAME),
       HYPERCOREF: os.path.join(os.getcwd(), HYPERCOREF, OUTPUT_FOLDER_NAME),
       ECBPLUS_METAM: os.path.join(os.getcwd(), ECBPLUS_METAM, OUTPUT_FOLDER_NAME),
}

EXCL_DATASETS_CONLL = [NEWSWCL50, HYPERCOREF, FCC]


def form_export_uCDCR_dataset():
    output_folder = os.path.join(os.getcwd(), "uCDCR")

    for dataset, folder in tqdm(DIRECTORIES_TO_OUTPUT.items(), desc="Copying datasets"):
        dataset_clean = dataset.replace("-prep", "")
        dataset_output_folder = os.path.join(output_folder, dataset_clean)
        os.makedirs(dataset_output_folder, exist_ok=True)

        # copy dataset readme
        src = os.path.join(folder, "..", "README.md")
        dst = os.path.join(dataset_output_folder, "README.md")
        shutil.copy2(src, dst)

        for name in os.listdir(folder):

            dst = os.path.join(dataset_output_folder, name)
            src = os.path.join(folder, name)
            if name.endswith(".parquet"):
                # the documents which are not public don't make into uCDCR in a conll-like form
                if name == "all_documents.parquet" and dataset in EXCL_DATASETS_CONLL:
                    continue

                # copy parquet
                shutil.copy2(src, dst)

            elif os.path.isdir(src):
                # copy mention files keeping the split folders
                os.makedirs(dst, exist_ok=True)
                for mention_file in os.listdir(src):
                    if not mention_file.endswith(".json"):
                        continue

                    src_mention = os.path.join(src, mention_file)
                    dst_mention = os.path.join(dst, mention_file)
                    shutil.copy2(src_mention, dst_mention)

    # upload to huggingface
    # upload_folder(folder_path=output_folder, repo_id="AnZhu/uCDCR", repo_type="dataset")
    LOGGER.info("uCDCR is exported!")


if __name__ == '__main__':
    check_datasets(DIRECTORIES_TO_OUTPUT)
    form_export_uCDCR_dataset()
