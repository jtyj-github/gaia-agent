"""
Official GAIA Scorer.
Implements the exact scoring logic used by HuggingFace GAIA benchmark.
"""

import re
import string
import warnings


def normalize_number_str(number_str: str) -> float:
    """
    Normalize a number string by removing common units and commas.

    Args:
        number_str: String representation of a number

    Returns:
        Float value, or infinity if conversion fails
    """
    # Replace common units and commas to allow conversion to float
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        print(f"String {number_str} cannot be normalized to number str.")
        return float("inf")


def split_string(
    s: str,
    char_list: list[str] = [",", ";"],
) -> list[str]:
    """
    Split string by comma or semicolon.

    Args:
        s: String to split
        char_list: List of separator characters

    Returns:
        List of split strings
    """
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    """
    Normalize a string by:
    - Removing all white spaces
    - Optionally removing punctuation (if remove_punct is True)
    - Converting to lowercase

    Args:
        input_str: The string to normalize
        remove_punct: Whether to remove punctuation (default: True)

    Returns:
        The normalized string
    """
    # Remove all white spaces. Required e.g for seagull vs. sea gull
    no_spaces = re.sub(r"\s", "", input_str)

    # Remove punctuation, if specified.
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()


def is_float(element: any) -> bool:
    """
    Check if an element can be converted to float.

    Args:
        element: Element to check

    Returns:
        True if convertible to float, False otherwise
    """
    try:
        float(element)
        return True
    except (ValueError, TypeError):
        return False


def question_scorer(
    model_answer: str,
    ground_truth: str,
) -> bool:
    """
    Score a model answer against ground truth using GAIA benchmark rules.

    Args:
        model_answer: The model's predicted answer
        ground_truth: The correct answer

    Returns:
        True if correct, False otherwise
    """
    if model_answer is None:
        model_answer = "None"

    # If gt is a number
    if is_float(ground_truth):
        print(f"Evaluating {model_answer} as a number.")
        normalized_answer = normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth)

    # If gt is a list
    elif any(char in ground_truth for char in [",", ";"]):
        print(f"Evaluating {model_answer} as a comma separated list.")

        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        # Check length is the same
        if len(gt_elems) != len(ma_elems):
            warnings.warn(
                "Answer lists have different lengths, returning False.", UserWarning
            )
            return False

        # Compare each element as float or str
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                # We do not remove punct since comparisons can include punct
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False)
                    == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    # If gt is a str
    else:
        print(f"Evaluating {model_answer} as a string.")
        return normalize_str(model_answer) == normalize_str(ground_truth)
