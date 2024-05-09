import re


def is_english(s: str) -> bool:
    try:
        s.encode(encoding="utf-8").decode("ascii")
    except UnicodeDecodeError:
        return False
    else:
        return True


def is_standard(s: str) -> bool:
    # check if there is cyrillic characters (U+0400..U+04FF)
    return not bool(re.search("[а-яА-Я]", s))


def main() -> None:
    print(is_english("Hello"))
    print(is_english("こんにちは"))
    print(is_english("Привет"))
    print(is_english("你好"))
    print(is_english("안녕하세요"))
    print(is_english("’"))


if __name__ == "__main__":
    main()
