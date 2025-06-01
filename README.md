# ChatAI z zamianą R na L

Prosta aplikacja webowa z chatbotem, który zamienia literę "R" na "L" w swoich odpowiedziach.

## Wymagania

- Python 3.7 lub nowszy
- pip (menedżer pakietów Pythona)

## Instalacja lokalna

1. Sklonuj to repozytorium:
```bash
git clone [URL_REPOZYTORIUM]
cd [NAZWA_KATALOGU]
```

2. Zainstaluj wymagane pakiety:
```bash
pip install -r requirements.txt
```

3. Uruchom aplikację:
```bash
python app.py
```

4. Otwórz przeglądarkę i przejdź pod adres:
```
http://localhost:5000
```

## Wdrożenie na Render.com (darmowy hosting)

1. Utwórz konto na [Render.com](https://render.com) (darmowe)

2. Po zalogowaniu, kliknij "New +" i wybierz "Web Service"

3. Połącz swoje repozytorium GitHub z Render

4. Wybierz repozytorium z tą aplikacją

5. Wypełnij formularz wdrożenia:
   - Name: chatai-r-to-l (lub inna nazwa)
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`

6. Kliknij "Create Web Service"

7. Render automatycznie wdroży Twoją aplikację i poda Ci URL, pod którym będzie dostępna

## Uwagi

- Aplikacja używa darmowego modelu językowego OPT-125M od Facebooka
- Wszystkie odpowiedzi chatbota będą miały zamienioną literę "R" na "L"
- Aplikacja może działać zarówno lokalnie jak i w internecie
- Render.com oferuje darmowy plan hostingowy z pewnymi ograniczeniami 