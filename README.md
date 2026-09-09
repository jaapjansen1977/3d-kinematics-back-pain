# Bukbewegingsanalyse

Streamlit-webapp voor de klinische en bewegingswetenschappelijke analyse van herhaald bukken. De gebruiker uploadt een Excelbestand met kinematische hoeken; de app segmenteert de herhalingen en toont ROM, tempo, golfvormstabiliteit, heup-rugkoppeling, beweging buiten het sagittale vlak en exploratieve complexiteitsmaten.

## Functies

- Upload van `.xlsx` met controle op verplichte kolommen en tijdas
- Automatische segmentatie van het ingestelde aantal bukherhalingen
- ROM van knie, heup, bekken en rug per herhaling
- Buigduur, strekduur, cyclustijd en verblijftijd onderin
- Golfvormovereenkomst, opeenvolgende RMSE, vector coding en CRP
- Lateroflexie, romprotatie en laterale bekkenbeweging
- Sample Entropy, SPARC, spectrale verhouding, PCA en bewaakte DFA
- Vraagteken-tooltips bij KPI's en tabelkolommen
- Download van alle resultaten als ZIP met CSV, JSON en PNG

## Excelopmaak

Het standaardwerkblad heet `angles`. Verplichte kolommen:

```text
frame, time_s, knee_flex_deg_L, knee_flex_deg_R,
hip_flex_deg_L, hip_flex_deg_R, pelvis_flex_deg,
trunk_flex_deg, pelvis_abd_deg, trunk_abd_deg, trunk_rot_deg
```

Optioneel: `hip_abd_deg_L`, `hip_abd_deg_R`, `hip_rot_deg_L`, `hip_rot_deg_R`.

Een fictief voorbeeldbestand staat in `example_data/`.

## Lokaal starten

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Op Windows PowerShell is de activatie-opdracht `.venv\\Scripts\\Activate.ps1`.

## Testen

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Publiceren via GitHub en Streamlit Community Cloud

1. Maak een nieuwe GitHub-repository en plaats alle bestanden uit deze map in de repository.
2. Meld je aan op [Streamlit Community Cloud](https://share.streamlit.io/) met GitHub.
3. Kies **Create app**, selecteer de repository en branch, en gebruik `app.py` als entrypoint.
4. Publiceer de app. Wijzigingen worden daarna via nieuwe GitHub-commits uitgerold.

`requirements.txt` staat bewust in de hoofdmap, naast `app.py`, zodat Community Cloud alle Python-afhankelijkheden installeert.

## Privacy

De app schrijft uploads alleen naar een tijdelijke map tijdens de analyse en verwijdert die map daarna automatisch. De resultaten blijven in de actieve Streamlit-sessie beschikbaar. Een publieke Community Cloud-app is alleen bedoeld voor fictieve of volledig geanonimiseerde data. Gebruik voor herleidbare gezondheidsgegevens een afgeschermde, juridisch en organisatorisch passende omgeving met toegangscontrole, verwerkersafspraken, loggingbeleid en een vastgestelde bewaartermijn.

## Methodologische beperkingen

- `lumbar_flex_rel_deg` is een planaire benadering: `trunk_flex_deg - pelvis_flex_deg`.
- Relatieve axiale romp-bekkenrotatie is niet beschikbaar zonder globale bekkenrotatie.
- Uitkomsten zijn beschrijvend; de app gebruikt geen klinische afkapwaarden zonder geschikte norm- en betrouwbaarheidsdata.
- Sample Entropy, SPARC en spectrale maten zijn exploratief en gevoelig voor datalengte en preprocessing.
- DFA wordt alleen berekend bij minimaal 600 cycli.

De wetenschappelijke bronnen en hun toepassing staan in `analysis.py` en worden in de app getoond onder **Methode & export**.

## Licentie

MIT. Zie `LICENSE`.
