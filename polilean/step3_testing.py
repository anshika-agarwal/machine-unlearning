import sys
import os

def choice(agree, disagree, threshold=0.3):
    """
    Returns an integer code based on agree/disagree:
      0 => Strongly Disagree
      1 => Disagree
      2 => Agree
      3 => Strongly Agree
    """
    if agree == 0 and disagree == 0:
        return 1  # Default "Disagree" if both are zero
    if agree >= disagree + threshold:
        return 3  # Strongly Agree
    elif agree >= disagree:
        return 2  # Agree
    elif disagree >= agree + threshold:
        return 0  # Strongly Disagree
    else:
        return 1  # Disagree

# Map numeric codes to text labels
code_to_text = {
    0: "Strongly Disagree",
    1: "Disagree",
    2: "Agree",
    3: "Strongly Agree"
}

def main(input_file, output_file, threshold=0.3):
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        statement_index = 0
        for line in fin:
            line = line.strip()
            if not line:
                continue
            # Example line: "0 agree: 0.34699 disagree: 0.65300"
            parts = line.split()

            # Parse float values
            agree_val = float(parts[2])
            disagree_val = float(parts[4])

            # Compute the integer code (0..3)
            code_val = choice(agree_val, disagree_val, threshold)
            # Convert to a text label
            label = code_to_text[code_val]

            # Write output (number + text label)
            fout.write(
                f"Statement {statement_index}: agree={agree_val:.3f}, "
                f"disagree={disagree_val:.3f} => code={code_val} ({label})\n"
            )
            statement_index += 1

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python step3_testing.py <input_file> <output_file> [threshold]")
        print("  Example: python step3_testing.py score/Llama-3.2-1B.txt results/Llama-3.2-1B.txt 0.3")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    # Optional threshold (default=0.3)
    threshold = float(sys.argv[3]) if len(sys.argv) == 4 else 0.3

    main(input_file, output_file, threshold)
    print(f"Parsing complete. Results written to: {output_file}")