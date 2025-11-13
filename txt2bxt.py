import re

from fire import Fire


def text_to_number_list(text):
    text = re.sub(" |\t|\n|", "", text)
    result = text.split(",")
    return [int(i) for i in result if i != ""]


def mod_number_list(number_list, mod):
    return [i % mod for i in number_list]


def number_to_binary_list(number_list, bit_length):
    return [bin(i)[2:].zfill(bit_length) for i in number_list]


def str_list_to_file(file_path, str_list):
    with open(file_path, "w") as f:
        for str in str_list:
            f.write(str + "\n")


def write_llm_output_to_file(file_path, llm_output, mod=256, bit_length=8):
    number_list = text_to_number_list(llm_output)
    moded_number_list = mod_number_list(number_list, mod)
    binary_list = number_to_binary_list(moded_number_list, bit_length)
    str_list_to_file(file_path, binary_list)


def convert_text_file_to_binary_text_file(text_file_path, binary_text_file_path, mod=256, bit_length=8):
    with open(text_file_path, "r") as f:
        llm_output = f.read()
    write_llm_output_to_file(binary_text_file_path, llm_output, mod, bit_length)


if __name__ == "__main__":
    Fire(convert_text_file_to_binary_text_file)
