"""Per-language curated prompt sets for the multilingual web demo.

Each list parallels the Rust `CUSTOM_PROBLEMS` in `extra_prompts.py`: small,
stdlib-only function-completion problems with a single function signature
and a docstring/comment describing the intent. The LLM is asked to
complete the body, so each prompt ends right after the opening `{` (Java,
C++) or after the `def` line + opening (Python).

Why these problems: short enough to fit within the wall budget but long
enough to cross multiple boundary checkpoints once expanded. Examples
are intentionally analogous to the Rust LeetCode-style set ("reverse a
list", "FizzBuzz", "Count vowels") so the per-language behaviour stays
comparable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CustomProblem:
    name: str        # dropdown id
    title: str       # one-line dropdown label
    prompt: str      # the source the model is asked to complete


# ─── Java ─────────────────────────────────────────────────────────────


_JAVA_REVERSE = """\
/**
 * Reverse a list of integers in place.
 *
 * Given a List<Integer>, swap elements end-to-end so the first becomes
 * the last, etc. No new list is allocated; the input is mutated.
 *
 * Example:
 *   Input:  [1, 2, 3, 4, 5]
 *   After:  [5, 4, 3, 2, 1]
 *
 * Constraints:
 *   - list may be empty
 *   - list.size() up to 1000
 */
import java.util.List;

public class Main {
    public static void reverseList(List<Integer> list) {
"""


_JAVA_FIZZBUZZ = """\
/**
 * Classic FizzBuzz.
 *
 * For each integer from 1 to n inclusive, append to the result list:
 *   - "FizzBuzz" if i is divisible by both 3 and 5
 *   - "Fizz"     if i is divisible by 3 only
 *   - "Buzz"     if i is divisible by 5 only
 *   - the decimal string of i otherwise
 *
 * Example:
 *   Input:  n = 5
 *   Output: ["1", "2", "Fizz", "4", "Buzz"]
 *
 * Constraints:
 *   - 0 <= n <= 10000
 *   - return an empty list when n == 0
 */
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static List<String> fizzBuzz(int n) {
"""


_JAVA_COUNT_VOWELS = """\
/**
 * Count the number of vowels in a string.
 *
 * Vowels are the five characters a, e, i, o, u — case-insensitive.
 * The string may contain any Unicode character; non-ASCII letters do
 * not count as vowels here.
 *
 * Example:
 *   Input:  "Hello, World!"
 *   Output: 3   (e, o, o)
 *
 * Constraints:
 *   - s.length() up to 100000
 */
public class Main {
    public static int countVowels(String s) {
"""


JAVA_CUSTOM_PROBLEMS: list[CustomProblem] = [
    CustomProblem(
        name="java_reverse_list",
        title="Java: Reverse a List<Integer> in place.",
        prompt=_JAVA_REVERSE,
    ),
    CustomProblem(
        name="java_fizzbuzz",
        title="Java: Classic FizzBuzz over 1..n returning a List<String>.",
        prompt=_JAVA_FIZZBUZZ,
    ),
    CustomProblem(
        name="java_count_vowels",
        title="Java: Count the vowels in a string (case-insensitive).",
        prompt=_JAVA_COUNT_VOWELS,
    ),
]


# ─── C++ ──────────────────────────────────────────────────────────────


_CPP_REVERSE = """\
// Reverse a std::vector<int> in place.
//
// Given a vector, swap elements end-to-end so the first becomes the
// last, etc. No new vector is allocated; the input is mutated.
//
// Example:
//   Input:  {1, 2, 3, 4, 5}
//   After:  {5, 4, 3, 2, 1}
//
// Constraints:
//   - vec may be empty
//   - vec.size() up to 1000
#include <vector>

void reverseVec(std::vector<int>& vec) {
"""


_CPP_FIZZBUZZ = """\
// Classic FizzBuzz.
//
// For each integer from 1 to n inclusive, append to the result vector:
//   - "FizzBuzz" if i is divisible by both 3 and 5
//   - "Fizz"     if i is divisible by 3 only
//   - "Buzz"     if i is divisible by 5 only
//   - std::to_string(i) otherwise
//
// Example:
//   Input:  n = 5
//   Output: {"1", "2", "Fizz", "4", "Buzz"}
//
// Constraints:
//   - 0 <= n <= 10000
//   - return an empty vector when n == 0
#include <string>
#include <vector>

std::vector<std::string> fizzBuzz(int n) {
"""


_CPP_COUNT_VOWELS = """\
// Count the number of vowels in a std::string.
//
// Vowels are the five characters a, e, i, o, u — case-insensitive.
// Non-ASCII characters do not count.
//
// Example:
//   Input:  "Hello, World!"
//   Output: 3   (e, o, o)
//
// Constraints:
//   - s.size() up to 100000
#include <string>

int countVowels(const std::string& s) {
"""


CPP_CUSTOM_PROBLEMS: list[CustomProblem] = [
    CustomProblem(
        name="cpp_reverse_vec",
        title="C++: Reverse a std::vector<int> in place.",
        prompt=_CPP_REVERSE,
    ),
    CustomProblem(
        name="cpp_fizzbuzz",
        title="C++: FizzBuzz over 1..n returning a vector<string>.",
        prompt=_CPP_FIZZBUZZ,
    ),
    CustomProblem(
        name="cpp_count_vowels",
        title="C++: Count vowels in a std::string (case-insensitive).",
        prompt=_CPP_COUNT_VOWELS,
    ),
]


# ─── Python ───────────────────────────────────────────────────────────


_PY_REVERSE = """\
def reverse_list(xs: list[int]) -> None:
    \"\"\"Reverse a list of integers in place.

    Swap elements end-to-end so the first becomes the last, etc. No new
    list is allocated; the input is mutated. Return None.

    Example:
        >>> xs = [1, 2, 3, 4, 5]
        >>> reverse_list(xs)
        >>> xs
        [5, 4, 3, 2, 1]
    \"\"\"
"""


_PY_FIZZBUZZ = """\
def fizzbuzz(n: int) -> list[str]:
    \"\"\"Classic FizzBuzz.

    For each integer from 1 to n inclusive, append to the result list:
      - "FizzBuzz" if i is divisible by both 3 and 5
      - "Fizz"     if i is divisible by 3 only
      - "Buzz"     if i is divisible by 5 only
      - str(i)     otherwise

    Returns an empty list when n == 0.

    Example:
        >>> fizzbuzz(5)
        ['1', '2', 'Fizz', '4', 'Buzz']
    \"\"\"
"""


_PY_COUNT_VOWELS = """\
def count_vowels(s: str) -> int:
    \"\"\"Count the vowels in a string.

    Vowels are the five characters a, e, i, o, u — case-insensitive.
    Non-ASCII letters do not count as vowels.

    Example:
        >>> count_vowels("Hello, World!")
        3
    \"\"\"
"""


PYTHON_CUSTOM_PROBLEMS: list[CustomProblem] = [
    CustomProblem(
        name="py_reverse_list",
        title="Python: Reverse a list[int] in place.",
        prompt=_PY_REVERSE,
    ),
    CustomProblem(
        name="py_fizzbuzz",
        title="Python: FizzBuzz over 1..n returning list[str].",
        prompt=_PY_FIZZBUZZ,
    ),
    CustomProblem(
        name="py_count_vowels",
        title="Python: Count vowels in a string (case-insensitive).",
        prompt=_PY_COUNT_VOWELS,
    ),
]


# ─── Dispatch ─────────────────────────────────────────────────────────


# (language, list of CustomProblem).
# "rust" intentionally NOT here — the rust list lives in `extra_prompts.py`
# alongside `load_rust_humaneval()`. The server merges them.
PROBLEMS_BY_LANG: dict[str, list[CustomProblem]] = {
    "java": JAVA_CUSTOM_PROBLEMS,
    "cpp": CPP_CUSTOM_PROBLEMS,
    "python": PYTHON_CUSTOM_PROBLEMS,
}
