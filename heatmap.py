import math
import os

import matplotlib.pyplot as plt
from ls_mlkit.util.sniffer import Sniffer


class Heatmap:
    def __init__(self, base=2):
        self.base = base

    def generate_heatmap(self, file_path: str, n: int = None, save_path=None):
        with open(file_path, "r") as file:
            data = file.read()
        # Count occurrences of each digit
        digit_counts = {str(i): 0 for i in range(self.base)}

        for char in data:
            if char.isdigit():
                digit_counts[char] += 1
        str_len = 0
        for i in range(self.base):
            str_len += digit_counts[str(i)]
        print(digit_counts)
        if n is None:
            n = math.sqrt(str_len)
            n = int(n)
        n2 = n * n
        # Extract n^2 digits and create n x n matrix
        matrix = []
        digit_list = []
        for char in data:
            if char.isdigit():
                digit_list.append(int(char))
                if len(digit_list) == n2:
                    break
        # Convert to n x n matrix
        for i in range(n):
            row = digit_list[i * n : (i + 1) * n]
            matrix.append(row)

        # Plot the heatmap using matplotlib
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, cmap="Reds")
        plt.colorbar()
        plt.title(f"{n}x{n} Digit Heatmap")
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.close()

    def generate_heatmap_from_dir(self, dir_path: str, pattern="bin.txt", n=None, save_dir=None):
        sniffer = Sniffer()
        file_path_list = sniffer.sniff_file(dir_path, pattern, -1)
        for file_path in file_path_list:
            print(f"dealing with {file_path}")
            file_name = os.path.basename(file_path)
            file_name = file_name.replace(".txt", "")
            if n is not None:
                file_name = file_name + f"_{n}x{n}"
            file_name = file_name + ".png"
            print(file_name)
            self.generate_heatmap(file_path, n=n, save_path=os.path.join(save_dir, file_name))


if __name__ == "__main__":
    heatmap = Heatmap()
    heatmap.generate_heatmap_from_dir("text", "bin.txt", save_dir="image", n=200)
    heatmap.generate_heatmap_from_dir("text", "bin.txt", save_dir="image")
