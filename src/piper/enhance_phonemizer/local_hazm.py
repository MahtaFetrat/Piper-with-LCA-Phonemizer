from typing import List, Optional

__all__ = ["stopwords_list", "Lemmatizer"]


_DEFAULT_STOPWORDS = sorted({
    "آیا", "آن", "آنها", "آن‌چه", "آن‌که", "آهای", "آی", "از", "اساسا", "است", "اش", "اگر", "اما", "امروز", "امسال",
    "امشب", "ان", "اند", "انشاءالله", "اول", "ای", "ایا", "اید", "ایشان", "ایم", "این", "اینجا", "اینک", "اینکه",
    "این‌که", "با", "باد", "بار", "بارة", "باره", "باش", "باشد", "باشند", "باشی", "باشید", "باشیم", "بالا", "بالای",
    "باید", "بدون", "بر", "برای", "برخی", "بسیار", "بسیاری", "بعد", "بعضی", "بله", "بلی", "بود", "بودن", "بودند",
    "بوده", "بوده‌اند", "بوده‌است", "بودی", "بودید", "بودیم", "بویژه", "بی", "بیا", "بیایید", "بیاییم", "بین", "ت",
    "تا", "تاکنون", "تان", "تحت", "تر", "ترین", "تمام", "توی", "تو", "توسط", "تول", "تک", "ثانیا", "جا", "جای", "جایی",
    "جدا", "جهت", "حاشیه", "حالا", "حتما", "حتی", "حداکثر", "حدود", "خارج", "خانم", "خانه", "خدا", "خواست", "خواه",
    "خواهد", "خواهند", "خواهی", "خواهید", "خواهیم", "خوب", "خود", "خودت", "خودتان", "خودش", "خودشان", "خودمان",
    "خوشبختانه", "خیلی", "داد", "دادن", "دار", "دارد", "دارند", "داری", "دارید", "داریم", "داشت", "داشتن", "داشتند",
    "داشته", "داشته‌باشد", "داشته‌باشند", "داشتی", "داشتید", "داشتیم", "دان", "دانست", "دانستن", "در", "درباره",
    "درون", "دریغ", "دقیقا", "دلا", "دلیل", "دم", "دنبال", "دو", "دوباره", "دوم", "دیروز", "دیشب", "دیگر", "دیگران",
    "دیگه", "را", "راه", "راحت", "رد", "رو", "روز", "روزانه", "روی", "ریز", "زیاد", "زیر", "زیرا", "سابق", "سپاس",
    "سراسر", "سری", "سریع", "سمت", "سوی", "سه", "سهم", "سی", "ش", "شا", "شاید", "شایسته", "شد", "شدن", "شدند", "شده",
    "شده‌", "شده‌اند", "شده‌بود", "شدی", "شدید", "شدیم", "شش", "شما", "شناس", "شنید", "شنیدن", "شود", "شوند", "شوی",
    "شوید", "شویم", "صاحب", "صدا", "صد", "صفحه", "صورت", "ط", "طرف", "طوری", "طول", "طی", "عالی", "عدم", "عقب", "علت",
    "علیه", "عمدا", "عمدتا", "عنوان", "غ", "غالب", "غالبا", "ف", "فراوان", "فردا", "فعلا", "فقط", "فکر", "فوق", "ق",
    "قبل", "قبلا", "قصد", "قوی", "ك", "كا", "كار", "كاش", "كاملا", "كجا", "كرد", "كردن", "كرده", "كس", "كسانی", "كسی",
    "كل", "كم", "كماكان", "كن", "كند", "كنند", "كنونی", "كه", "كو", "كی", "گ", "گا", "گاه", "گاهی", "گذاشت", "گذاشتن",
    "گذشته", "گرد", "گردد", "گرفت", "گرفتن", "گرفته", "گرم", "گروه", "گفت", "گفتن", "گفته", "گو", "گوید", "گویند",
    "گه", "گی", "ل", "لا", "لابد", "لازم", "لحظه", "لطفا", "م", "ما", "مان", "ماند", "ماندن", "مانند", "مبادا", "متاسفانه",
    "متوجه", "مثل", "مجددا", "مدام", "مدت", "مدتی", "مردم", "مرسی", "مستقیم", "مسلما", "مسئله", "مطمئنا", "معمولا",
    "مقابل", "مگر", "ممکن", "من", "موارد", "مورد", "موقع", "مگر", "می", "میان", "می‌باشد", "می‌باشند", "می‌شود",
    "می‌شوند", "می‌کنیم", "می‌کنند", "می‌کنی", "می‌کنید", "ن", "نا", "ناشی", "نام", "نباید", "نبود", "نخست", "ندارد",
    "ندارند", "نداشت", "نداشته", "نداشته‌باشد", "نزد", "نزدیک", "نشد", "نشده", "نشان", "نشانی", "نظر", "نظیر", "نکرده",
    "نما", "نماید", "نمود", "نمودن", "نموده", "نه", "نوزده", "نوشت", "نوشتن", "نویس", "نی", "نیز", "نیست", "نیستند",
    "نیم", "و", "وا", "وای", "واقعا", "وجود", "وگرنه", "ور", "وسط", "وضع", "وقت", "وقتی", "ولی", "وی", "ه", "ها",
    "های", "هایی", "هر", "هرچه", "هرگز", "هزار", "هست", "هستند", "هستیم", "هفت", "هم", "همان", "همه", "همواره",
    "همیشه", "همین", "همچنان", "همچون", "هی", "هیچ", "هیچگاه", "ی", "یا", "یاد", "یازده", "یک", "یکی"
})


_MINIMAL_VERBS = [
    "بود#باش", "شد#شو", "کرد#کن", "داشت#دار", "گفت#گو", "آمد#آ", "رفت#رو",
    "زد#زن", "گرفت#گیر", "ماند#مان", "دید#بین", "خواست#خواه", "توانست#توان",
    "خورد#خور", "آورد#آور", "داد#ده", "گذاشت#گذار", "گذشت#گذر", "برد#بر",
    "یافت#یاب", "شناخت#شناس", "رسید#رس", "کشید#کش", "شد#شو", "هست#باش"
]


def stopwords_list(stopwords_file: Optional[str] = None) -> List[str]:
    if stopwords_file:
        try:
            with open(stopwords_file, encoding="utf8") as f:
                return sorted({w.strip() for w in f})
        except Exception:
            return _DEFAULT_STOPWORDS
    return _DEFAULT_STOPWORDS


class Stemmer:
    def __init__(self) -> None:
        self.ends = [
            "ات", "ان", "ترین", "تر", "م", "ت", "ش", "یی", "ی", "ها", "ٔ", "‌ا", "‌",
        ]

    def stem(self, word: str) -> str:
        for end in self.ends:
            if word.endswith(end):
                word = word[:-len(end)]

        if word.endswith("ۀ"):
            word = word[:-1] + "ه"

        return word


class Conjugation:
    def perfective_past(self, ri: str) -> List[str]:
        return [ri + x for x in ["م", "ی", "", "یم", "ید", "ند"]]

    def negative_perfective_past(self, ri: str) -> List[str]:
        return ["ن" + x for x in self.perfective_past(ri)]

    def passive_perfective_past(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.perfective_past("شد")]

    def negative_passive_perfective_past(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.negative_perfective_past("شد")]

    def imperfective_past(self, ri: str) -> List[str]:
        return ["می‌" + x for x in self.perfective_past(ri)]

    def negative_imperfective_past(self, ri: str) -> List[str]:
        return ["ن" + x for x in self.imperfective_past(ri)]

    def passive_imperfective_past(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.imperfective_past("شد")]

    def negative_passive_imperfective_past(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.negative_imperfective_past("شد")]

    def past_progresive(self, ri: str) -> List[str]:
        return [x + " " + y for x, y in zip(self.perfective_past("داشت"), self.imperfective_past(ri))]

    def passive_past_progresive(self, ri: str) -> List[str]:
        return [x + " " + y for x, y in zip(self.perfective_past("داشت"), self.passive_imperfective_past(ri))]

    def present_perfect(self, ri: str) -> List[str]:
        return [ri + x for x in ["ه‌ام", "ه‌ای", "ه است", "ه", "ه‌ایم", "ه‌اید", "ه‌اند"]]

    def negative_present_perfect(self, ri: str) -> List[str]:
        return ["ن" + x for x in self.present_perfect(ri)]

    def subjunctive_present_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.perfective_present("باش")]

    def negative_subjunctive_present_perfect(self, ri: str) -> List[str]:
        return ["ن" + x for x in self.subjunctive_present_perfect(ri)]

    def grammatical_present_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + ("باش" if x == "باشی" else x) for x in self.perfective_present("باش")]

    def negative_grammatical_present_perfect(self, ri: str) -> List[str]:
        return ["ن" + ri + "ه " + ("باش" if x == "باشی" else x) for x in self.perfective_present("باش")]

    def passive_present_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.present_perfect("شد")]

    def negative_passive_present_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.negative_present_perfect("شد")]

    def passive_subjunctive_present_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.subjunctive_present_perfect("شد")]

    def negative_passive_subjunctive_present_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.negative_subjunctive_present_perfect("شد")]

    def passive_grammatical_present_perfect(self, ri: str) -> List[str]:
        return [ri + "ه شده " + ("باش" if x == "باشی" else x) for x in self.perfective_present("باش")]

    def negative_passive_grammatical_present_perfect(self, ri: str) -> List[str]:
        return [ri + "ه نشده " + ("باش" if x == "باشی" else x) for x in self.perfective_present("باش")]

    def imperfective_present_perfect(self, ri: str) -> List[str]:
        return ["می‌" + x for x in self.present_perfect(ri)]

    def negative_imperfective_present_perfect(self, ri: str) -> List[str]:
        return ["ن" + x for x in self.imperfective_present_perfect(ri)]

    def subjunctive_imperfective_present_perfect(self, ri: str) -> List[str]:
        return ["می‌" + x for x in self.subjunctive_present_perfect(ri)]

    def negative_subjunctive_imperfective_present_perfect(self, ri: str) -> List[str]:
        return ["ن" + x for x in self.subjunctive_imperfective_present_perfect(ri)]

    def passive_imperfective_present_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.imperfective_present_perfect("شد")]

    def negative_passive_imperfective_present_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.negative_imperfective_present_perfect("شد")]

    def passive_subjunctive_imperfective_present_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.subjunctive_imperfective_present_perfect("شد")]

    def negative_passive_subjunctive_imperfective_present_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + "ن" + x for x in self.subjunctive_imperfective_present_perfect("شد")]

    def present_perfect_progressive(self, ri: str) -> List[str]:
        return [x + " " + y for x, y in zip(self.present_perfect("داشت"), self.imperfective_present_perfect(ri))]

    def passive_present_perfect_progressive(self, ri: str) -> List[str]:
        return [x + " " + y for x, y in zip(self.present_perfect("داشت"), self.passive_imperfective_present_perfect(ri))]

    def past_precedent(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.perfective_past("بود")]

    def negative_past_precedent(self, ri: str) -> List[str]:
        return ["ن" + x for x in self.past_precedent(ri)]

    def passive_past_precedent(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.past_precedent("شد")]

    def negative_passive_past_precedent(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.negative_past_precedent("شد")]

    def imperfective_past_precedent(self, ri: str) -> List[str]:
        return ["می‌" + x for x in self.past_precedent(ri)]

    def negative_imperfective_past_precedent(self, ri: str) -> List[str]:
        return ["ن" + x for x in self.imperfective_past_precedent(ri)]

    def passive_imperfective_past_precedent(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.imperfective_past_precedent("شد")]

    def negative_passive_imperfective_past_precedent(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.negative_imperfective_past_precedent("شد")]

    def past_precedent_progressive(self, ri: str) -> List[str]:
        return [x + " " + y for x, y in zip(self.perfective_past("داشت"), self.imperfective_past_precedent(ri))]

    def passive_past_precedent_progressive(self, ri: str) -> List[str]:
        return [x + " " + y for x, y in zip(self.perfective_past("داشت"), self.passive_imperfective_past_precedent(ri))]

    def past_precedent_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.present_perfect("بود")]

    def negative_past_precedent_perfect(self, ri: str) -> List[str]:
        return ["ن" + x for x in self.past_precedent_perfect(ri)]

    def subjunctive_past_precedent_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.subjunctive_present_perfect("بود")]

    def negative_subjunctive_past_precedent_perfect(self, ri: str) -> List[str]:
        return ["ن" + x for x in self.subjunctive_past_precedent_perfect(ri)]

    def grammatical_past_precedent_perfect(self, ri: str) -> List[str]:
        return [ri + "ه بوده " + ("باش" if x == "باشی" else x) for x in self.perfective_present("باش")]

    def negative_grammatical_past_precedent_perfect(self, ri: str) -> List[str]:
        return ["ن" + x for x in self.grammatical_past_precedent_perfect(ri)]

    def passive_past_precedent_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.past_precedent_perfect("شد")]

    def negative_passive_past_precedent_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.negative_past_precedent_perfect("شد")]

    def passive_subjunctive_past_precedent_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.subjunctive_past_precedent_perfect("شد")]

    def negative_passive_subjunctive_past_precedent_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + "ن" + x for x in self.subjunctive_past_precedent_perfect("شد")]

    def passive_grammatical_past_precedent_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.grammatical_past_precedent_perfect("شد")]

    def negative_passive_grammatical_past_precedent_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.negative_grammatical_past_precedent_perfect("شد")]

    def imperfective_past_precedent_perfect(self, ri: str) -> List[str]:
        return ["می‌" + x for x in self.past_precedent_perfect(ri)]

    def negative_imperfective_past_precedent_perfect(self, ri: str) -> List[str]:
        return ["ن" + x for x in self.imperfective_past_precedent_perfect(ri)]

    def subjunctive_imperfective_past_precedent_perfect(self, ri: str) -> List[str]:
        return ["می‌" + x for x in self.subjunctive_past_precedent_perfect(ri)]

    def negative_subjunctive_imperfective_past_precedent_perfect(self, ri: str) -> List[str]:
        return ["ن" + x for x in self.subjunctive_imperfective_past_precedent_perfect(ri)]

    def passive_imperfective_past_precedent_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.imperfective_past_precedent_perfect("شد")]

    def negative_passive_imperfective_past_precedent_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.negative_imperfective_past_precedent_perfect("شد")]

    def passive_subjunctive_imperfective_past_precedent_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.subjunctive_imperfective_past_precedent_perfect("شد")]

    def negative_passive_subjunctive_imperfective_past_precedent_perfect(self, ri: str) -> List[str]:
        return [ri + "ه " + "ن" + x for x in self.subjunctive_imperfective_past_precedent_perfect("شد")]

    def past_precedent_perfect_progressive(self, ri: str) -> List[str]:
        return [x + " " + y for x, y in zip(self.present_perfect("داشت"), self.imperfective_past_precedent_perfect(ri))]

    def passive_past_precedent_perfect_progressive(self, ri: str) -> List[str]:
        return [x + " " + y for x, y in zip(self.present_perfect("داشت"), self.passive_imperfective_past_precedent_perfect(ri))]

    def perfective_present(self, rii: str) -> List[str]:
        return [rii + x for x in ["م", "ی", "د", "یم", "ید", "ند"]]

    def negative_perfective_present(self, rii: str) -> List[str]:
        return ["ن" + x for x in self.perfective_present(rii)]

    def subjunctive_perfective_present(self, rii: str) -> List[str]:
        return ["ب" + x for x in self.perfective_present(rii)]

    def negative_subjunctive_perfective_present(self, rii: str) -> List[str]:
        return ["ن" + x for x in self.perfective_present(rii)]

    def grammatical_perfective_present(self, rii: str) -> List[str]:
        return ["ببین" if x == "ببینی" else x for x in self.subjunctive_perfective_present(rii)]

    def negative_grammatical_perfective_present(self, rii: str) -> List[str]:
        return ["ن" + ("بین" if x == "بینی" else x) for x in self.perfective_present(rii)]

    def passive_perfective_present(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.perfective_present("شو")]

    def negative_passive_perfective_present(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.negative_perfective_present("شو")]

    def passive_subjunctive_perfective_present(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.subjunctive_perfective_present("شو")]

    def negative_passive_subjunctive_perfective_present(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.negative_subjunctive_perfective_present("شو")]

    def passive_grammatical_perfective_present(self, ri: str) -> List[str]:
        return [ri + "ه " + ("بشو" if x == "بشوی" else x) for x in self.grammatical_perfective_present("شو")]

    def negative_passive_grammatical_perfective_present(self, ri: str) -> List[str]:
        return [ri + "ه " + ("نشو" if x == "نشوی" else x) for x in self.negative_grammatical_perfective_present("شو")]

    def imperfective_present(self, rii: str) -> List[str]:
        return ["می‌" + x for x in self.perfective_present(rii)]

    def negative_imperfective_present(self, rii: str) -> List[str]:
        return ["ن" + x for x in self.imperfective_present(rii)]

    def passive_imperfective_present(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.imperfective_present("شو")]

    def negative_passive_imperfective_present(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.negative_imperfective_present("شو")]

    def present_progressive(self, rii: str) -> List[str]:
        return [x + " " + y for x, y in zip(self.perfective_present("دار"), self.imperfective_present(rii))]

    def passive_present_progressive(self, ri: str) -> List[str]:
        return [x + " " + y for x, y in zip(self.perfective_present("دار"), self.passive_imperfective_present(ri))]

    def perfective_future(self, ri: str) -> List[str]:
        return [x + " " + ri for x in self.perfective_present("خواه")]

    def negative_perfective_future(self, ri: str) -> List[str]:
        return ["ن" + x for x in self.perfective_future(ri)]

    def passive_perfective_future(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.perfective_future("شد")]

    def negative_passive_perfective_future(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.negative_perfective_future("شد")]

    def imperfective_future(self, ri: str) -> List[str]:
        return ["می‌" + x for x in self.perfective_future(ri)]

    def negative_imperfective_future(self, ri: str) -> List[str]:
        return ["ن" + x for x in self.imperfective_future(ri)]

    def passive_imperfective_future(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.imperfective_future("شد")]

    def negative_passive_imperfective_future(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.negative_imperfective_future("شد")]

    def future_precedent(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.perfective_future("بود")]

    def negative_future_precedent(self, ri: str) -> List[str]:
        return ["ن" + x for x in self.future_precedent(ri)]

    def passive_future_precedent(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.future_precedent("شد")]

    def negative_passive_future_precedent(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.negative_future_precedent("شد")]

    def future_precedent_imperfective(self, ri: str) -> List[str]:
        return ["می‌" + x for x in self.future_precedent(ri)]

    def negative_future_precedent_imperfective(self, ri: str) -> List[str]:
        return ["ن" + x for x in self.future_precedent_imperfective(ri)]

    def passive_future_precedent_imperfective(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.future_precedent_imperfective("شد")]

    def negative_passive_future_precedent_imperfective(self, ri: str) -> List[str]:
        return [ri + "ه " + x for x in self.negative_future_precedent_imperfective("شد")]

    def get_all(self, verb: str) -> List[str]:
        ri, rii = verb.split("#")
        infinitive = [ri + "ن"]
        result = [infinitive]
        result.append(self.perfective_past(ri))
        result.append(self.negative_perfective_past(ri))
        result.append(self.passive_perfective_past(ri))
        result.append(self.negative_passive_perfective_past(ri))
        result.append(self.imperfective_past(ri))
        result.append(self.negative_imperfective_past(ri))
        result.append(self.passive_imperfective_past(ri))
        result.append(self.negative_passive_imperfective_past(ri))
        result.append(self.past_progresive(ri))
        result.append(self.passive_past_progresive(ri))
        result.append(self.present_perfect(ri))
        result.append(self.negative_present_perfect(ri))
        result.append(self.subjunctive_present_perfect(ri))
        result.append(self.negative_subjunctive_present_perfect(ri))
        result.append(self.grammatical_present_perfect(ri))
        result.append(self.negative_grammatical_present_perfect(ri))
        result.append(self.passive_present_perfect(ri))
        result.append(self.negative_passive_present_perfect(ri))
        result.append(self.passive_subjunctive_present_perfect(ri))
        result.append(self.negative_passive_subjunctive_present_perfect(ri))
        result.append(self.passive_grammatical_present_perfect(ri))
        result.append(self.negative_passive_grammatical_present_perfect(ri))
        result.append(self.imperfective_present_perfect(ri))
        result.append(self.negative_imperfective_present_perfect(ri))
        result.append(self.subjunctive_imperfective_present_perfect(ri))
        result.append(self.negative_subjunctive_imperfective_present_perfect(ri))
        result.append(self.passive_imperfective_present_perfect(ri))
        result.append(self.negative_passive_imperfective_present_perfect(ri))
        result.append(self.passive_subjunctive_imperfective_present_perfect(ri))
        result.append(self.negative_passive_subjunctive_imperfective_present_perfect(ri))
        result.append(self.present_perfect_progressive(ri))
        result.append(self.passive_present_perfect_progressive(ri))
        result.append(self.past_precedent(ri))
        result.append(self.negative_past_precedent(ri))
        result.append(self.passive_past_precedent(ri))
        result.append(self.negative_passive_past_precedent(ri))
        result.append(self.imperfective_past_precedent(ri))
        result.append(self.negative_imperfective_past_precedent(ri))
        result.append(self.passive_imperfective_past_precedent(ri))
        result.append(self.negative_passive_imperfective_past_precedent(ri))
        result.append(self.past_precedent_progressive(ri))
        result.append(self.passive_past_precedent_progressive(ri))
        result.append(self.past_precedent_perfect(ri))
        result.append(self.negative_past_precedent_perfect(ri))
        result.append(self.subjunctive_past_precedent_perfect(ri))
        result.append(self.negative_subjunctive_past_precedent_perfect(ri))
        result.append(self.grammatical_past_precedent_perfect(ri))
        result.append(self.negative_grammatical_past_precedent_perfect(ri))
        result.append(self.passive_past_precedent_perfect(ri))
        result.append(self.negative_passive_past_precedent_perfect(ri))
        result.append(self.passive_subjunctive_past_precedent_perfect(ri))
        result.append(self.negative_passive_subjunctive_past_precedent_perfect(ri))
        result.append(self.passive_grammatical_past_precedent_perfect(ri))
        result.append(self.negative_passive_grammatical_past_precedent_perfect(ri))
        result.append(self.imperfective_past_precedent_perfect(ri))
        result.append(self.negative_imperfective_past_precedent_perfect(ri))
        result.append(self.subjunctive_imperfective_past_precedent_perfect(ri))
        result.append(self.negative_subjunctive_imperfective_past_precedent_perfect(ri))
        result.append(self.passive_imperfective_past_precedent_perfect(ri))
        result.append(self.negative_passive_imperfective_past_precedent_perfect(ri))
        result.append(self.passive_subjunctive_imperfective_past_precedent_perfect(ri))
        result.append(self.negative_passive_subjunctive_imperfective_past_precedent_perfect(ri))
        result.append(self.past_precedent_perfect_progressive(ri))
        result.append(self.passive_past_precedent_perfect_progressive(ri))
        result.append(self.perfective_present(rii))
        result.append(self.negative_perfective_present(rii))
        result.append(self.subjunctive_perfective_present(rii))
        result.append(self.negative_subjunctive_perfective_present(rii))
        result.append(self.grammatical_perfective_present(rii))
        result.append(self.negative_grammatical_perfective_present(rii))
        result.append(self.passive_perfective_present(ri))
        result.append(self.negative_passive_perfective_present(ri))
        result.append(self.passive_subjunctive_perfective_present(ri))
        result.append(self.negative_passive_subjunctive_perfective_present(ri))
        result.append(self.passive_grammatical_perfective_present(ri))
        result.append(self.negative_passive_grammatical_perfective_present(ri))
        result.append(self.imperfective_present(rii))
        result.append(self.negative_imperfective_present(rii))
        result.append(self.passive_imperfective_present(ri))
        result.append(self.negative_passive_imperfective_present(ri))
        result.append(self.present_progressive(rii))
        result.append(self.passive_present_progressive(ri))
        result.append(self.perfective_future(ri))
        result.append(self.negative_perfective_future(ri))
        result.append(self.passive_perfective_future(ri))
        result.append(self.negative_passive_perfective_future(ri))
        result.append(self.imperfective_future(ri))
        result.append(self.negative_imperfective_future(ri))
        result.append(self.passive_imperfective_future(ri))
        result.append(self.negative_passive_imperfective_future(ri))
        result.append(self.future_precedent(ri))
        result.append(self.negative_future_precedent(ri))
        result.append(self.passive_future_precedent(ri))
        result.append(self.negative_passive_future_precedent(ri))
        result.append(self.future_precedent_imperfective(ri))
        result.append(self.negative_future_precedent_imperfective(ri))
        result.append(self.passive_future_precedent_imperfective(ri))
        result.append(self.negative_passive_future_precedent_imperfective(ri))
        return sum(result, [])


class Lemmatizer:
    def __init__(
        self,
        words_file: Optional[str] = None,
        verbs_file: Optional[str] = None,
        joined_verb_parts: bool = True,
    ) -> None:
        self.words = {}
        self.verbs = {}
        self.stemmer = Stemmer()
        self.conjugation = Conjugation()

        if words_file:
            try:
                with open(words_file, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 1:
                            self.words[parts[0]] = parts[0]
            except Exception:
                pass

        raw_verbs = _MINIMAL_VERBS
        if verbs_file:
            try:
                with open(verbs_file, encoding="utf-8") as f:
                    raw_verbs = [line.strip() for line in f if line.strip()]
            except Exception:
                pass

        self.verbs["است"] = "#است"

        after_verbs = {
            "ام", "ای", "است", "ایم", "اید", "اند", "بودم", "بودی", "بود", "بودیم", "بودید", "بودند",
            "باشم", "باشی", "باشد", "باشیم", "باشید", "باشند", "شده_ام", "شده_ای", "شده_است", "شده_ایم",
            "شده_اید", "شده_اند", "شده_بودم", "شده_بودی", "شده_بود", "شده_بودیم", "شده_بودید", "شده_بودند",
            "شده_باشم", "شده_باشی", "شده_باشد", "شده_باشیم", "شده_باشید", "شده_باشند", "نشده_ام", "نشده_ای",
            "نشده_است", "نشده_ایم", "نشده_اید", "نشده_اند", "نشده_بودم", "نشده_بودی", "نشده_بود", "نشده_بودیم",
            "نشده_بودید", "نشده_بودند", "نشده_باشم", "نشده_باشی", "نشده_باشد", "نشده_باشیم", "نشده_باشید",
            "نشده_باشند", "شوم", "شوی", "شود", "شویم", "شوید", "شوند", "شدم", "شدی", "شد", "شدیم", "شدید",
            "شدند", "نشوم", "نشوی", "نشود", "نشویم", "نشوید", "نشوند", "نشدم", "نشدی", "نشد", "نشدیم", "نشدید",
            "نشدند", "می‌شوم", "می‌شوی", "می‌شود", "می‌شویم", "می‌شوید", "می‌شوند", "می‌شدم", "می‌شدی", "می‌شد",
            "می‌شدیم", "می‌شدید", "می‌شدند", "نمی‌شوم", "نمی‌شوی", "نمی‌شود", "نمی‌شویم", "نمی‌شوید", "نمی‌شوند",
            "نمی‌شدم", "نمی‌شدی", "نمی‌شد", "نمی‌شدیم", "نمی‌شدید", "نمی‌شدند", "خواهم_شد", "خواهی_شد", "خواهد_شد",
            "خواهیم_شد", "خواهید_شد", "خواهند_شد", "نخواهم_شد", "نخواهی_شد", "نخواهد_شد", "نخواهیم_شد",
            "نخواهید_شد", "نخواهند_شد",
        }
        before_verbs = {
            "خواهم", "خواهی", "خواهد", "خواهیم", "خواهید", "خواهند",
            "نخواهم", "نخواهی", "نخواهد", "نخواهیم", "نخواهید", "نخواهند",
        }

        for verb in raw_verbs:
            for tense in self.conjugation.get_all(verb):
                self.verbs[tense] = verb

            if joined_verb_parts:
                bon = verb.split("#")[0]
                for after_verb in after_verbs:
                    self.verbs[bon + "ه_" + after_verb] = verb
                    self.verbs["ن" + bon + "ه_" + after_verb] = verb
                for before_verb in before_verbs:
                    self.verbs[before_verb + "_" + bon] = verb

    def lemmatize(self, word: str, pos: str = "") -> str:
        if not pos and word in self.words:
            return word

        if (not pos or pos == "VERB") and word in self.verbs:
            return self.verbs[word]

        if pos.startswith("ADJ") and word[-1] == "ی":
            return word

        if pos == "PRON":
            return word

        if word in self.words:
            return word

        stem = self.stemmer.stem(word)
        if stem and stem in self.words:
            return stem

        return word
