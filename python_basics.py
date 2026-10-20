"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set("aeiouAEIOU")
    return sum(1 for ch in text if ch in vowels)

def test_count_vowels():
    assert count_vowels(TextInput("hello")) == 2
    assert count_vowels(TextInput("AEIOU")) == 5
    assert count_vowels(TextInput("xyz")) == 0
    assert count_vowels(TextInput("")) == 0
    assert count_vowels(TextInput("aAeEiIoOuU")) == 10
    assert count_vowels(TextInput("bcdfg")) == 0
    assert count_vowels(TextInput("Programming")) == 3
    print("1 accept")

test_count_vowels()

def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))

def test_has_unique_characters():
    assert has_unique_characters(TextInput("abc")) is True
    assert has_unique_characters(TextInput("aab")) is False
    assert has_unique_characters(TextInput("")) is True
    assert has_unique_characters(TextInput("a")) is True
    assert has_unique_characters(TextInput("abcdefghij")) is True
    assert has_unique_characters(TextInput("abcdefghija")) is False
    assert has_unique_characters(TextInput("Aa")) is True  # регистр важен
    assert has_unique_characters(TextInput("  ")) is False  # пробелы считаются  
    print("2 accept")  

test_has_unique_characters()

def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")

def test_count_one_bits():
    assert count_one_bits(PositiveIntegerInput(0b1011)) == 3
    assert count_one_bits(PositiveIntegerInput(1)) == 1
    assert count_one_bits(PositiveIntegerInput(255)) == 8
    assert count_one_bits(PositiveIntegerInput(0)) == 0
    assert count_one_bits(PositiveIntegerInput(1024)) == 1
    assert count_one_bits(PositiveIntegerInput(2**32 - 1)) == 32
    assert count_one_bits(PositiveIntegerInput(7)) == 3
    print("3 accept")    

test_count_one_bits()

def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps

def test_multiplicative_persistence():
    assert multiplicative_persistence(PositiveIntegerInput(39)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(4)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(999)) == 4
    assert multiplicative_persistence(PositiveIntegerInput(10)) == 1
    assert multiplicative_persistence(PositiveIntegerInput(77)) == 4
    assert multiplicative_persistence(PositiveIntegerInput(277777788888899)) == 11
    assert multiplicative_persistence(PositiveIntegerInput(1)) == 0
    print("4 accept")

test_multiplicative_persistence()    

def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    n = len(predicted)
    total = sum((p - e) ** 2 for p, e in zip(predicted, expected))
    return total / n

def test_mse():
    assert mse(VectorPairInput([1, 2], [1, 2])) == 0.0
    assert mse(VectorPairInput([0, 0], [1, 1])) == 1.0
    assert mse(VectorPairInput([1], [2])) == 1.0
    assert mse(VectorPairInput([1, 2, 3], [1, 2, 3])) == 0.0
    assert mse(VectorPairInput([0, 0, 0], [1, 1, 1])) == 1.0
    assert mse(VectorPairInput([1, 2], [3, 4])) == 4.0
    assert mse(VectorPairInput([0, 0, 0, 0], [1, 1, 1, 1])) == 1.0
    print("5 accept")

test_mse()

def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = []
    divisor = 2
    while divisor * divisor <= number:
        if number % divisor == 0:
            power = 0
            while number % divisor == 0:
                number //= divisor
                power += 1
            factors.append((divisor, power))
        divisor += 1
    if number > 1:
        factors.append((number, 1))

    result = ""
    for base, power in factors:
        if power == 1:
            result += f"({base})"
        else:
            result += f"({base}**{power})"
    return result    

def test_prime_factorization():
    assert prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(PositiveIntegerInput(7)) == "(7)"
    assert prime_factorization(PositiveIntegerInput(12)) == "(2**2)(3)"
    assert prime_factorization(PositiveIntegerInput(2)) == "(2)"
    assert prime_factorization(PositiveIntegerInput(1)) == ""
    assert prime_factorization(PositiveIntegerInput(4)) == "(2**2)"
    assert prime_factorization(PositiveIntegerInput(100)) == "(2**2)(5**2)"
    assert prime_factorization(PositiveIntegerInput(97)) == "(97)"
    print("6 accept")

test_prime_factorization()

def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0
    while total < cube_count:
        k += 1
        total += k * k
    if total == cube_count:
        return k
    return "It is impossible"

def test_pyramid():
    assert pyramid(PositiveIntegerInput(14)) == 3
    assert pyramid(PositiveIntegerInput(5)) == 2
    assert pyramid(PositiveIntegerInput(1)) == 1
    assert pyramid(PositiveIntegerInput(2)) == "It is impossible"
    assert pyramid(PositiveIntegerInput(30)) == 4
    assert pyramid(PositiveIntegerInput(55)) == 5
    assert pyramid(PositiveIntegerInput(3)) == "It is impossible"
    assert pyramid(PositiveIntegerInput(100)) == "It is impossible"
    print("7 accept")

test_pyramid()

def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = [int(d) for d in str(number)]
    n = len(digits)

    if n % 2 == 1:
        mid = n // 2
        left = digits[:mid]
        right = digits[mid + 1:]
    else:
        mid = n // 2
        left = digits[:mid - 1]
        right = digits[mid + 1:]

    return sum(left) == sum(right)
    
def test_is_balanced_number():
    assert is_balanced_number(PositiveIntegerInput(1234006)) is True
    assert is_balanced_number(PositiveIntegerInput(123456)) is False
    assert is_balanced_number(PositiveIntegerInput(11)) is True
    assert is_balanced_number(PositiveIntegerInput(101)) is True
    assert is_balanced_number(PositiveIntegerInput(1001)) is True
    assert is_balanced_number(PositiveIntegerInput(1230)) is False
    assert is_balanced_number(PositiveIntegerInput(123321)) is True
    assert is_balanced_number(PositiveIntegerInput(12345)) is False
    assert is_balanced_number(PositiveIntegerInput(0)) is True
    assert is_balanced_number(PositiveIntegerInput(7)) is True 
    print("8 accept")

test_is_balanced_number()   
