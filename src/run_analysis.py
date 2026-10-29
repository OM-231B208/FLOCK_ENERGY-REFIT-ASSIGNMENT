"""Run Sections 1–3 of the REFIT analysis in assignment order."""

from section1_raw_cleaning import run as run_section1
from section2_washing_machine_eda import run as run_section2
from section3_disaggregation import run as run_section3


def main() -> None:
    print("\n## 1. Basic inspection and cleaning of raw data")
    run_section1()
    print("\n## 2. Exploratory analysis of washing-machine use")
    run_section2()
    print("\n## 3. Build and evaluate an appliance-disaggregation model")
    run_section3()


if __name__ == "__main__":
    main()
