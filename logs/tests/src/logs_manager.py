import re


## FUNCTIONS
def print_match_per_accuracy(file_path, accuracy_target):
    pattern = (r"TESTING:\n"
               r"[\s\S]*?--> Final Accuracy: ([0-9.]+)\n"
               r"[\s\S]*?--> Final Loss: ([0-9.]+)\n"
               r"[\s\S]*?--> Final Precision: ([0-9.]+)\n"
               r"[\s\S]*?--> Final Recall: ([0-9.]+)\n"
               r"[\s\S]*?--> Final F1 Score: ([0-9.]+)\n"
               r"HYPERPARAMETERS:\n"
               r"[\s\S]*?--> Hidden Size: (\[.*?\]), Learning Rate: ([0-9.]+), "
               r"Batch Size: ([0-9]+), Alpha: ([0-9.]+), Dropout: ([0-9.]+), "
               r"Weight Decay: ([0-9.]+)")

    try:
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            load = file.read()
            matches = re.finditer(pattern, load, re.DOTALL)
            for match in matches:
                final_accuracy = float(match.group(1))
                if final_accuracy == accuracy_target:
                    print(match.group(0))
    except FileNotFoundError:
        print("ERROR: file not found")
    except Exception as e:
        print(f"ERROR occurred while reading: {e}")


## MAIN
if __name__ == "__main__":
    file_path = '../GENE EXPRESSION & METHYLATION STATISTICS & OS - Binary Classification/Torch Trials/MLP (GPU) - 93,75 - 48 features.txt'
    accuracy_target = 0.8125
    print_match_per_accuracy(file_path, accuracy_target)
