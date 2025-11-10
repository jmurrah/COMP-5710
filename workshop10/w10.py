"""
COMP-5710 Workshop 10: Simplistic White-box Fuzzing
Author: Jacob Murrah
Date: 11/11/2025
"""


def divide(v1, v2):
    return v1 / v2


def simpleFuzzer():
    errors = [
        ("a", 2),  # String numerator (error #1)
        (2, "b"),  # String denominator (error #2)
        ("a", "b"),  # Both string (error #3)
        (2, 0),  # Division by zero (error #4)
        (None, 2),  # None numerator (error #5)
        (2, None),  # None denominator (error #6)
        (None, None),  # Both None (error #7)
        ([2], 2),  # List numerator (error #8)
        (2, {"a": 2}),  # Dict denominator (error #9)
        (set([1, 2]), set([3, 4])),  # Both set (error #10)
    ]

    output_filename = "crash_messages.txt"
    with open(output_filename, "w") as f:
        for i, error in enumerate(errors):
            try:
                result = divide(error[0], error[1])
                message = f"divide({error[0]}, {error[1]}) = {result}"
            except Exception as e:
                message = f"divide({error[0]}, {error[1]}) raised an exception: {e}"
            f.write(f"Error {i+1}: {message}\n")

    print(f"Fuzzing complete. Results written to {output_filename}")


if __name__ == "__main__":
    simpleFuzzer()
