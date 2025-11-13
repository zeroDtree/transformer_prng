import re

import numpy as np
from ls_mlkit.util.sniffer import Sniffer


class ReportToCSV:
    def __init__(self):
        pass

    def convert_report_to_csv(self, report_path: str, csv_path: str = ""):
        with open(report_path, "r") as f:
            report = f.readlines()
        pattern = r"^\s*(?P<c_number>(\d+\s+){10})\d+\.\d+\s+\d+/\d+\s+\w+$"
        np_list = []
        for line in report:
            match = re.match(pattern, line)
            if match:
                c_numbers = match.group("c_number")
                c_numbers_array = np.fromstring(c_numbers, sep=" ", dtype=int)
                np_list.append(c_numbers_array)
        numpy_array = np.array(np_list)
        sumed_array = numpy_array.sum(axis=0)
        print(sumed_array)

    def convert_report_dir_to_csv(
        self,
        report_dir: str,
        csv_path: str = "",
        pattern="final",
    ):
        sniffer = Sniffer()
        file_path_list = sniffer.sniff_file(report_dir, pattern, -1)
        for file_path in file_path_list:
            print(f"converting {file_path}")
            self.convert_report_to_csv(file_path, csv_path)


if __name__ == "__main__":
    report_to_csv = ReportToCSV()
    report_to_csv.convert_report_dir_to_csv("./text")
