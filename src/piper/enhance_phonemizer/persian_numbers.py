import re


DIGITS_MAP = {
    '0': 'صفر', '1': 'یک', '2': 'دو', '3': 'سه', '4': 'چهار',
    '5': 'پنج', '6': 'شش', '7': 'هفت', '8': 'هشت', '9': 'نه'
}

TENS = {
    10: 'ده', 11: 'یازده', 12: 'دوازده', 13: 'سیزده', 14: 'چهارده',
    15: 'پانزده', 16: 'شانزده', 17: 'هفده', 18: 'هجده', 19: 'نوزده',
    20: 'بیست', 30: 'سی', 40: 'چهل', 50: 'پنجاه',
    60: 'شصت', 70: 'هفتاد', 80: 'هشتاد', 90: 'نود'
}

HUNDREDS = {
    100: 'صد', 200: 'دویست', 300: 'سیصد', 400: 'چهارصد', 500: 'پانصد',
    600: 'ششصد', 700: 'هفتصد', 800: 'هشتصد', 900: 'نهصد'
}


def _convert_three_digit(num: int) -> str:
    if num == 0:
        return ''

    if num < 10:
        return DIGITS_MAP[str(num)]
    elif num < 20:
        return TENS[num]
    elif num < 100:
        tens_part = (num // 10) * 10
        ones_part = num % 10
        if ones_part == 0:
            return TENS[tens_part]
        return f"{TENS[tens_part]} و {DIGITS_MAP[str(ones_part)]}"
    else:
        hundreds_part = (num // 100) * 100
        rem = num % 100
        if rem == 0:
            return HUNDREDS[hundreds_part]
        return f"{HUNDREDS[hundreds_part]} و {_convert_three_digit(rem)}"


def num_to_text(num: int) -> str:
    if num == 0:
        return 'صفر'

    if num < 0:
        return f"منفی {num_to_text(abs(num))}"

    if num < 1000:
        return _convert_three_digit(num)

    parts = []

    if num >= 1_000_000_000:
        billions = num // 1_000_000_000
        parts.append(f"{_convert_three_digit(billions)} میلیارد")
        num %= 1_000_000_000

    if num >= 1_000_000:
        millions = num // 1_000_000
        parts.append(f"{_convert_three_digit(millions)} میلیون")
        num %= 1_000_000

    if num >= 1000:
        thousands = num // 1000
        parts.append(f"{_convert_three_digit(thousands)} هزار")
        num %= 1000

    if num > 0:
        parts.append(_convert_three_digit(num))

    return ' و '.join(parts)


def _read_phone_chunk(chunk: str) -> str:
    if not chunk:
        return ""

    if all(c == '0' for c in chunk):
        count = len(chunk)
        if count == 2:
            return "دو صفر"
        elif count == 3:
            return "سِتا صفر"
        elif count == 4:
            return "چهارتا صفر"
        else:
            return f"{num_to_text(count)} تا صفر"

    result_parts = []
    temp_chunk = chunk

    while temp_chunk.startswith('0'):
        result_parts.append("صفر")
        temp_chunk = temp_chunk[1:]

    if temp_chunk:
        val = int(temp_chunk)
        result_parts.append(num_to_text(val))

    return " ".join(result_parts)


def _smart_split_phone(phone_str: str, has_plus: bool = False) -> list:
    length = len(phone_str)
    chunks = []

    if has_plus:
        if phone_str.startswith('98') and len(phone_str) > 5:
            chunks.append("+" + phone_str[:2])
            rest = phone_str[2:]
            if rest.startswith('9'):

                inner_chunks = _smart_split_phone("0" + rest)
                chunks.extend(inner_chunks)
                return chunks
            else:
                chunks.append(rest)
                return chunks

        elif phone_str.startswith('1') and length == 11:
            chunks.append("+" + phone_str[:1])
            chunks.append(phone_str[1:4])
            chunks.append(phone_str[4:7])
            chunks.append(phone_str[7:])
            return chunks

    if phone_str.startswith('09') and length == 11:
        chunks.append(phone_str[:4])
        rest = phone_str[4:]

        part_mid = rest[:3]
        part_end = rest[3:]

        is_end_round = False
        if part_end == '0000':
            is_end_round = True
        elif part_end.endswith('00'):
            is_end_round = True
        elif part_end[1] == '0' and part_end[2] == '0':
            is_end_round = True
        if part_mid == '000':
            is_end_round = True

        if is_end_round:
            chunks.append(part_mid)
            chunks.append(part_end)
        else:
            chunks.append(rest[:3])
            chunks.append(rest[3:5])
            chunks.append(rest[5:])
        return chunks

    if phone_str.startswith('0') and length == 11:
        chunks.append(phone_str[:3])
        rest = phone_str[3:]

        part1 = rest[:4]
        part2 = rest[4:]

        if (part1.endswith('00') and part2.endswith('00')) or (part2 == '0000'):
            chunks.append(part1)
            chunks.append(part2)
            return chunks

        p3_1 = rest[:3]
        p3_2 = rest[3:6]
        if p3_1.endswith('0') and p3_2.endswith('0'):
            chunks.append(p3_1)
            chunks.append(p3_2)
            chunks.append(rest[6:])
            return chunks

        chunks.append(rest[:2])
        chunks.append(rest[2:4])
        chunks.append(rest[4:6])
        chunks.append(rest[6:])
        return chunks

    if not phone_str.startswith('0'):
        if length == 8:
            chunks.append(phone_str[:2])
            chunks.append(phone_str[2:4])
            chunks.append(phone_str[4:6])
            chunks.append(phone_str[6:])
            return chunks
        elif length == 4:
            chunks.append(phone_str)
            return chunks
        elif length == 5:
            chunks.append(phone_str)
            return chunks

    if length == 10 and phone_str.startswith('9'):
        chunks.append(phone_str[:3])
        chunks.append(phone_str[3:6])
        chunks.append(phone_str[6:8])
        chunks.append(phone_str[8:])
        return chunks

    return [phone_str]


def phone_to_text(raw_input: str) -> str:
    clean_input = raw_input.replace(' ', '').replace(
        '-', '').replace('(', '').replace(')', '')

    persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    english_digits = '0123456789'
    trans_table = str.maketrans(persian_digits, english_digits)
    clean_input = clean_input.translate(trans_table)

    has_plus = False
    if clean_input.startswith('+'):
        has_plus = True
        clean_input = clean_input[1:]

    if not clean_input.isdigit():
        return raw_input

    chunks = _smart_split_phone(clean_input, has_plus)

    text_parts = []
    for ch in chunks:
        if ch.startswith('+'):
            val = int(ch[1:])
            text_parts.append(f"مثبت {num_to_text(val)}")
        else:
            text_parts.append(_read_phone_chunk(ch))

    return "، ".join(text_parts)


def _is_likely_phone(num_str: str) -> bool:
    if num_str.startswith('+'):
        return True

    if num_str.startswith('09') and len(num_str) == 11:
        return True

    if num_str.startswith('0') and len(num_str) >= 7:
        return True

    return False


def find_and_normalize_numbers(text: str) -> str:
    text = text.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789'))\
                .translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789'))

    pattern = r'(?:\+|-)?\d+(?:,\d+)*'

    def replace_match(match):
        original_str = match.group()
        clean_str = original_str.replace(',', '')

        if _is_likely_phone(clean_str):
            return phone_to_text(clean_str)
        else:
            try:
                val = int(clean_str)
                return num_to_text(val)
            except ValueError:
                return original_str

    return re.sub(pattern, replace_match, text)


if __name__ == "__main__":
    examples = [

        "شماره من ۰۹۱۲۳۴۵۶۷۸۹ است",
        "تلفن شرکت ۰۲۱۸۸۰۵۶۰۷۰ می باشد",
        "کد تایید: ۸۸۹۹۱۱۰۰",
        "تماس بین المللی: +۹۸۹۱۵۱۰۰۲۰۳۰",
        "شارژ مستقیم ۰۹۳۵۲۰۰۳۰۴۰",
        "کد پستی ۱۱۱۱۱۰۰۰۰۰",


        "قیمت این کالا ۵,۴۰۰ تومان است",
        "جمعیت ایران ۸۵۰۰۰۰۰۰ نفر است",
        "دمای هوا منفی ۵ درجه است: -5",
        "تعداد ۱۰۰۱ شب",
        "عدد صفر 0"
    ]

    print("--- بررسی عملکرد کد ادغام شده ---\n")
    for ex in examples:
        converted = find_and_normalize_numbers(ex)
        print(f"Original: {ex}")
        print(f"Converted: {converted}")
        print("-" * 30)
