import time
from zipfile import ZipFile
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Unzip and merge multiple chunked zips from a Google Drive download"
    )
    parser.add_argument(
        "--i",
        required=True,
        type=str,
        help="Directory containing the downloaded zips (ex. `C:\\Users\\...\\Downloads`)"
    )
    parser.add_argument(
        "--n",
        required=True,
        type=str,
        help="The name of the original directory (ex. `04_03_23_EC_1`)"
    )
    parser.add_argument(
        "--t",
        required=True,
        type=str,
        help="The download timestamp appended to each folder (ex. `20230706T020731Z`)"
    )
    args = parser.parse_args()

    i = 1
    start = time.time()

    while True:
        try:
            path = f"{args.i}/{args.n}-{args.t}-{str(i).zfill(3)}.zip"

            with ZipFile(path) as zipFile:
                zipFile.extractall(args.i)

            print(f"Extracted {path}")
            i += 1

        except FileNotFoundError:  # TODO: hacky?
            end = time.time()
            print(f"Finished extracting {i - 1} folders in {end - start}s")
            break
