Acesta este modeulul nostru de explainer AI pentru sah.

Codul din foldere poate fi rulat asa cum este avand deja un model antrenat.

Pentru a-l rula trebuie sa aveti instalat python3 si urmatoarele librarii:
- pytorch
- stockfish
- python-chess
- numpy
- flask
- flask_cors 

In urma rularii codului de python se poate deschide interfata grafica pentru a interactiona cu AI-ul.

Puteti sa puneti orice fel de intrebari la care doriti explicatie mai putin cele care incep cu:
"Avand urmatoarea partida de sah"
"Avand urmatoarea pozitie de sah"

Pe de alta parte, daca puneti intrebari care incep cu "Avand urmatoarea partdia de sah" urmata de o secventa valida de mutari de sah, si continuand co propozitie care sa cuprinda numarul de mutari recomandate pe care vrem sa le facem (un numar intre 1 si 9) atunci AI-ul va recomanda mutarile pe care le-ar face in acea situatie si va scoate si o probabilitate de castig.
(ex: "Avand urmatoarea partida e4 e5 care sunt urmatoarele cele mai bune 2 mutari?")

Atentie! Mutarile trebuie separate prin numai un spatiu

Daca puneti intrebari care incep cu "Avand urmatoarea pozitie de sah" urmata de o notatie fen valida si o propozitie care sa precizeze un numar intre 1 si 9 de mutari pe care le vrem in avans atunci AI-ul va recomanda mutarile pe care le-ar face in acea situatie si va scoate si o probabilitate de castig.
(ex: Avand urmatoarea pozitie rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2 care sunt cele mai bune 2 mutari in avans?)

Pentru a antrena reteaua neuronala trebuie sa decomentati codul din fisierul main si sa il comentati pe cel existent. De asemenea, in fisierul chat.txt trebuie sa se gaseasca o secventa de intrebari raspunsuri sub forma:
Q1
A1
Q2
A2
...
Atentie! Intrebarile si raspunsurile trebuie separate prin numai un spatiu

Sugestie de antrenare:
Luati toate intrebarile ce se regasesc in fieserul all-data.txt, copiati-le in chat.txt si incercati sa antrenati pana ajungeti la un loss de 0.001
