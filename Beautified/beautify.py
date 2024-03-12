import csv
from operator import index

FOLDER = "./Beautified"
DATASET_FOLDER = "../Datasets"

# Change this manually
DATASET_FILE = "ramp3.txt"
HEADER_ROWS = 5
DATARATE = 860
CSV_HEADER = [
    "Timestamp",
    "Log_number",
    "Value"
]

DATASET = f"{DATASET_FOLDER}/{DATASET_FILE}"

HEADER = f"{FOLDER}/{DATASET_FILE[0:-4]}-header.txt"
DATA = f"{FOLDER}/{DATASET_FILE[0:-4]}-data.csv"
ERRORS = f"{FOLDER}/{DATASET_FILE[0:-4]}-errors.txt"



with open(DATASET, "r", newline="") as dataset_handler:
    with open(HEADER, "w", newline= "") as header_handler:
        with open(DATA, "w", newline="") as beautified_dataset:
            with open(ERRORS, "w", newline="") as errors_file:

        
                for _ in range(0, HEADER_ROWS):
                    tmp = dataset_handler.readline()
                    header_handler.write(tmp)

                # Let's print the header for the CSV file:
                header = "\t".join(CSV_HEADER) + "\n"
                beautified_dataset.write(header)

                counter = 1

                for rows in dataset_handler:

                    print(f"Beautifying row number: {counter}\n")

                    items = rows.split(" ")

                    timestamp = items[0]

                    if len(items) == DATARATE + 1:

                        for i in range(1, len(items)):
                            if i == len(items) -1:
                                tmp = f"{timestamp}\t{i}\t{items[i][0:-1]}\n"
                            else:
                                tmp = f"{timestamp}\t{i}\t{items[i]}\n"
                            beautified_dataset.write(
                                tmp
                            )

                    else:

                        ORIGINAL_FILE_ERROR = HEADER_ROWS + counter

                        errors_file.write(
                            f"Error at line {ORIGINAL_FILE_ERROR} at timestamp {timestamp}.\n"
                            f"Found {len(items) - 1} samples\n\n"
                            f"NUMBERED DUMP:\n"
                        )

                        for i in range(1, len(items)):
                            errors_file.write(
                                f"{i}\t{items[i]}\n"
                            )



                    counter += 1


        
            
