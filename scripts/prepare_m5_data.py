import sys
sys.path.append("..")

from src.preprocessing import prepare_m5_data

if __name__ == "__main__":
    prepare_m5_data("../data/m5")
