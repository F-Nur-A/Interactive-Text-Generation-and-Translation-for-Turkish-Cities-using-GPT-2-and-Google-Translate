import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from googletrans import Translator
import csv
import torch

# Model ve tokenizer'ı yükleme
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Şehir verilerini yükleme
def load_city_data():
    iller = {}
    with open("iller.csv", newline='', encoding='cp1254') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader)
        for row in reader:
            if len(row) < 8:
                print("Hatalı satır:", row)
                continue
            il_adi = row[0]
            cografi_ozellikler = row[1].replace('|', ' ')
            tarih_ve_kultur = row[2].replace('|', ' ')
            dogal_guzellikler = row[3].replace('|', ' ')
            turistik_noktalar = row[4].replace('|', ' ')
            gastronomi = row[5].replace('|', ' ')
            alisveris = row[6].replace('|', ' ')
            etkinlikler_ve_festivaller = row[7].replace('|', ' ')
            il_bilgileri = f"{cografi_ozellikler} {tarih_ve_kultur} {dogal_guzellikler} {turistik_noktalar} Gastronomi: {gastronomi} {alisveris} |"
            iller[il_adi] = il_bilgileri
    return iller

def split_text_into_chunks(text, max_length):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return chunks

def generate_text():
    selected_city = city_combobox.get()
    if selected_city:
        city_info = city_data.get(selected_city)
        prompt = f"{selected_city}: {city_info}"
        input_chunks = split_text_into_chunks(prompt, 512)  # Maksimum uzunluğu 512 olarak değiştirdik

        generated_text = ""
        for chunk in input_chunks:
            input_ids = torch.tensor([chunk])
            attention_mask = torch.tensor([[1] * len(chunk)])
            try:
                output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
                chunk_generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                generated_text += chunk_generated_text
            except Exception as e:
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, f"Hata oluştu: {e}")
                return

        # Etkinlikler ve festivaller bilgisinden sonrasını kes
        index = generated_text.find("|")
        if index != -1:
            generated_text = generated_text[:index + len("|")]
            city_name, city_details = generated_text.split(": ", 1)  # Şehir adını ve detayları ayır
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"""
{city_name}
{city_details}""")
    else:
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Lütfen bir il seçiniz.")

translator = Translator()

def translate_text_to_language(text, language_code):
    if text.strip():
        try:
            translated_text = translator.translate(text, src="tr", dest=language_code).text
            return translated_text
        except Exception as e:
            return f"Hata oluştu: {e}"
    else:
        return "Lütfen önce bir metin oluşturun."

# Dil seçimi için fonksiyon
def select_language():
    selected_language = language_combobox.get()
    if selected_language:
        language_codes = {
            "İngilizce": "en",
            "Fransızca": "fr",
            "Almanca": "de",
            "İspanyolca": "es",
            "İtalyanca": "it"
        }
        language_code = language_codes.get(selected_language)
        generated_text = result_text.get(1.0, tk.END)
        translated_text = translate_text_to_language(generated_text, language_code)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, translated_text)

# Ana Tkinter uygulamasını oluşturma
root = tk.Tk()
root.title("TÜRKİYE’NİN 81 İLİ HAKKINDA METİN OLUŞTURMA UYGULAMASI")

# Şehir verilerini yükleme
city_data = load_city_data()

# Şehir seçimini içeren bir labelframe oluşturma
city_frame = ttk.LabelFrame(root, text="İl Seçimi")
city_frame.grid(column=0, row=0, padx=20, pady=20)

# Şehir seçimini içeren bir combobox oluşturma
city_label = ttk.Label(city_frame, text="İl:")
city_label.grid(column=0, row=0, padx=5, pady=5)
city_combobox = ttk.Combobox(city_frame, values=list(city_data.keys()), state="readonly")
city_combobox.grid(column=1, row=0, padx=5, pady=5)
city_combobox.current(0)

# Metin oluşturma düğmesi
generate_button = ttk.Button(city_frame, text="Metin Oluştur", command=generate_text)
generate_button.grid(column=0, row=1, columnspan=2, padx=5, pady=5)

# Sonuç gösterme alanı
result_text = scrolledtext.ScrolledText(root, width=150, height=30, wrap=tk.WORD)
result_text.grid(column=0, row=1, padx=10, pady=10)

# Dil seçimi için combobox
language_label = ttk.Label(city_frame, text="Çevrilecek Dil:")
language_label.grid(column=3, row=0, padx=5, pady=5)
language_combobox = ttk.Combobox(city_frame, values=["İngilizce", "Fransızca", "Almanca", "İspanyolca", "İtalyanca"], state="readonly")
language_combobox.grid(column=4, row=0, padx=5, pady=5)
language_combobox.current(0)

# Dil seçimi düğmesi
select_language_button = ttk.Button(city_frame, text="Çevir", command=select_language)
select_language_button.grid(column=4, row=1, padx=5, pady=5)

# Tkinter uygulamasını başlatma
root.mainloop()