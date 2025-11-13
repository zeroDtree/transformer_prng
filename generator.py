import torch
from fire import Fire
from ls_mlkit.util.decorators import timer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_text(
    model_path="./checkpoint/finetuned-mt19937_8-openai_community_gpt2",
    prompt="123",
    max_gen_len=256,
    do_sample=False,
):
    prompt = str(prompt)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    @timer()
    def generate_text(prompt, max_length=50, flag=True):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            do_sample=do_sample,
        )
        if flag:
            generated_text = outputs[0].tolist()
            tmp_str = ""
            for i in generated_text:
                i = i % 256
                tmp_str += str(i)
                tmp_str += ","
            generated_text = tmp_str
            print(generated_text)
        else:
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    generated_text = generate_text(prompt, max_length=max_gen_len)
    return generated_text[len(prompt) :]


def get_llm_output(model_path, num_sampling, num_bits):
    result = ""
    m = 2**num_bits
    for i in tqdm(range(num_sampling), desc="Generating LLM output"):
        prompt = str(f"{i%m}")
        generated_text = generate_text(model_path=model_path, prompt=prompt, max_gen_len=1000)
        result = result + "\n\n" + generated_text
    return result


def write_llm_output_to_file(file_path, llm_output):
    with open(file_path, "w") as f:
        f.write(llm_output)


def main(model_path, num_sampling, num_bits, file_path):
    llm_output = get_llm_output(model_path, num_sampling, num_bits)
    write_llm_output_to_file(file_path, llm_output)


if __name__ == "__main__":
    Fire(main)
