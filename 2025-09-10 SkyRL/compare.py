#!/usr/bin/env python3
# Minimal side-by-side generations from two models (no scoring).
# uv run --isolated --extra vllm -m compare

import os
from vllm import LLM, SamplingParams

# --- Hard-coded models ---
BASELINE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"                    # HF Hub ID
TRAINED_MODEL  = os.path.expanduser("~/exports/global_step_31/policy")  # Local HF export

# --- Prompts to test ---
PROMPTS = ["Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
    "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
    "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
    "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?",
    "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
    "Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?",
    "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?",
    "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?",
    "A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?",
    "Tobias is buying a new pair of shoes that costs $95. He has been saving up his money each month for the past three months. He gets a $5 allowance a month. He also mows lawns and shovels driveways. He charges $15 to mow a lawn and $7 to shovel. After buying the shoes, he has $15 in change. If he mows 4 lawns, how many driveways did he shovel?",
    "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?",
    "Jasper will serve charcuterie at his dinner party. He buys 2 pounds of cheddar cheese for $10, a pound of cream cheese that cost half the price of the cheddar cheese, and a pack of cold cuts that cost twice the price of the cheddar cheese. How much does he spend on the ingredients?",
    "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
]

# --- Generation settings ---
SAMPLING = SamplingParams(temperature=0.0, max_tokens=256)
TP = 1
GPU_UTIL = 0.9

def run(model_id, prompts):
    llm = LLM(model=model_id, tensor_parallel_size=TP, gpu_memory_utilization=GPU_UTIL)
    outs = llm.generate(prompts, SAMPLING)
    return [o.outputs[0].text for o in outs]

def write_txt(path, prompts, outputs):
    with open(path, "w", encoding="utf-8") as f:
        for i, (p, out) in enumerate(zip(prompts, outputs)):
            f.write(f"--- Prompt {i} ---\n{p}\n\n")
            f.write(out.strip() + "\n")
            f.write("=" * 80 + "\n")

def main():
    print("\n=== Baseline:", BASELINE_MODEL, "===\n")
    base = run(BASELINE_MODEL, PROMPTS)

    print("\n=== Trained:", TRAINED_MODEL, "===\n")
    trained = run(TRAINED_MODEL, PROMPTS)

    # Write to files for easy diff
    base_path = "baseline_outputs.txt"
    trained_path = "trained_outputs.txt"
    write_txt(base_path, PROMPTS, base)
    write_txt(trained_path, PROMPTS, trained)

    print(f"\nSaved baseline outputs to: {base_path}")
    print(f"Saved trained  outputs to: {trained_path}")

    # Optional: also print to stdout (you can remove this block if you want files only)
    for i, p in enumerate(PROMPTS):
        print(f"\n--- Prompt {i} ---\n{p}\n")
        print("Baseline output:\n" + base[i].strip() + "\n")
        print("Trained output:\n" + trained[i].strip() + "\n")
        print("=" * 80)

if __name__ == "__main__":
    main()
