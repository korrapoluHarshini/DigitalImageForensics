from pathlib import Path

DATASET_PATH = Path("datasets/CASIA2.0_revised")

AU_LIST = DATASET_PATH / "au_list.txt"
TP_LIST = DATASET_PATH / "tp_list.txt"


def inspect_list(file_path, name):
    print("=" * 60)
    print(f"{name}")
    print("=" * 60)

    if not file_path.exists():
        print(f"ERROR: {file_path} not found")
        return

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        lines = [line.strip() for line in file if line.strip()]

    print(f"Total entries: {len(lines)}")
    print("\nFirst 10 entries:")

    for i, line in enumerate(lines[:10], start=1):
        print(f"{i}. {line}")

    print()


inspect_list(AU_LIST, "AUTHENTIC IMAGE LIST")
inspect_list(TP_LIST, "TAMPERED IMAGE LIST")