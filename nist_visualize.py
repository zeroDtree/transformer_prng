import re

import numpy as np
from ls_mlkit.util.sniffer import Sniffer


class ReportHandler:
    def __init__(self):
        pass

    def extract_report(self, report_path: str):
        with open(report_path, "r") as f:
            report = f.readlines()
        pattern = r"^\s*(?P<c_number>(\d+\s+){10})\d+\.\d+\s+\d+/\d+\s+\w+$"
        lines = []
        for line in report:
            match = re.match(pattern, line)
            if match:
                lines.append(line)
        return lines

    def extract_c_numbers(self, lines: list):
        np_list = []
        pattern = r"^\s*(?P<c_number>(\d+\s+){10})\d+\.\d+\s+\d+/\d+\s+\w+$"
        for line in lines:
            match = re.match(pattern, line)
            if match:
                c_numbers = match.group("c_number")
                c_numbers_array = np.fromstring(c_numbers, sep=" ", dtype=int)
                np_list.append(c_numbers_array)
        numpy_array = np.array(np_list)
        return numpy_array

    def sum_array(self, numpy_array):
        sumed_array = numpy_array.sum(axis=0)
        return sumed_array

    def convert_report_dir_to_c_numbers_array(
        self,
        report_dir: str,
        pattern="final",
    ):
        sniffer = Sniffer()
        file_path_list = sniffer.sniff_file(report_dir, pattern, -1)
        for file_path in file_path_list:
            print(f"converting {file_path}")
            lines = self.extract_report(file_path)
            numpy_array = self.extract_c_numbers(lines)
            sumed_array = self.sum_array(numpy_array)
            return sumed_array

    def mean_p_value(self, lines: list):
        pattern = r"^\s*(\d+\s+){10}(?P<p_value>(\d+\.\d+))\s+\d+/\d+\s+(?P<test_name>\w+)$"
        info = {}
        for line in lines:
            match = re.match(pattern, line)
            if match:
                p_value = float(match.group("p_value"))
                test_name = match.group("test_name")
                if test_name not in info:
                    info[test_name] = {"p_value": 0, "count": 0}
                info[test_name]["p_value"] = info[test_name].get("p_value", 0) + p_value
                info[test_name]["count"] = info[test_name].get("count", 0) + 1

        for test_name, test_info in info.items():
            test_info["mean_p_value"] = test_info["p_value"] / test_info["count"]
        return info

    def format_print_p_value_info_for_latex(self, info: dict):
        text_list = []
        for test_name, test_info in info.items():
            text_list.append(f"({test_name},{test_info['mean_p_value']})")
        return "\n".join(text_list)

    def convert_report_dir_to_mean_p_value(self, report_dir: str):
        sniffer = Sniffer()
        file_path_list = sniffer.sniff_file(report_dir, "final", -1)
        for file_path in file_path_list:
            lines = self.extract_report(file_path)
            info = self.mean_p_value(lines)
            formatted_text = self.format_print_p_value_info_for_latex(info)
            with open(f"p_value_info.txt", "a") as f:
                f.write(file_path + "\n")
                f.write(formatted_text)
                f.write("\n\n")


if __name__ == "__main__":
    report_handler = ReportHandler()
    # report_handler.convert_report_dir_to_c_numbers_array("./text")
    report_handler.convert_report_dir_to_mean_p_value("./text")
