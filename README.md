# Predizione dei Consumi Energetici nell'Industria Siderurgica

Questo repository contiene un progetto di analisi dati e machine learning per prevedere il consumo energetico (`Usage_kWh`) in un'industria siderurgica. Il progetto è composto da due parti principali:

1.  Un **Jupyter Notebook** che guida attraverso l'intero processo di analisi, dalla pulizia dei dati all'addestramento e valutazione di diversi modelli predittivi.
2.  Una **Web App accessoria in Flask** che permette di interagire con il modello migliore generato dal notebook.

## Struttura del Repository

-   `Previsione Consumi Energetici w Web Flask.ipynb`: **(Punto di partenza)**. Il notebook completo che contiene tutta l'analisi esplorativa, la preparazione dei dati, il training dei modelli e la serializzazione degli artefatti finali.
-   `Steel_industry_data.csv`: Il dataset utilizzato per l'analisi.
-   `web-app/`: Una sottocartella contenente un'applicazione web Flask per testare il modello.
    -   `app.py`: Il codice sorgente dell'applicazione Flask.
    -   `templates/index.html`: Il template HTML per l'interfaccia utente.
    -   `requirements.txt`: Le dipendenze specifiche per la web app.

## Come Eseguire il Progetto

L'esecuzione del progetto avviene in due fasi sequenziali.

### Fase 1: Esecuzione del Jupyter Notebook

È **fondamentale** eseguire prima il notebook, poiché genera i file necessari al funzionamento della web app.

1.  **Prepara l'ambiente**: Installa le dipendenze necessarie per il notebook. Puoi usare il file `requirements.txt` fornito di seguito.
    ```bash
    pip install -r requirements.txt
    ```
2.  **Avvia Jupyter**: Esegui Jupyter Lab o Jupyter Notebook e apri il file `Previsione Consumi Energetici w Web Flask.ipynb`.
3.  **Esegui tutte le celle**: Esegui il notebook dall'inizio alla fine. L'ultima cella creerà i seguenti file nella directory principale:
    -   `energy_prediction_model.pkl` (il modello addestrato)
    -   `scaler.pkl` (il preprocessore per i dati numerici)
    -   `label_encoders.pkl` (i preprocessori per i dati categorici)
    -   `model_metadata.json` (metadati sul modello e sulle feature)
    -   `feature_ranges.json` (statistiche sulle feature per la web app)

### Fase 2: Avvio della Web App

Una volta che il notebook ha generato i file, puoi avviare l'applicazione web.

1.  **Sposta i file**: Copia i 5 file generati (`.pkl` e `.json`) dalla directory principale alla sottocartella `web-app/`.
2.  **Installa le dipendenze della web app**:
    ```bash
    pip install -r web-app/requirements.txt
    ```
3.  **Avvia l'applicazione**:
    ```bash
    python web-app/app.py
    ```
4.  **Apri il browser**: Visita l'indirizzo `http://localhost:5000` per interagire con il modello.

---
*Progetto di Programmazione di Applicazioni Data Intensive A.A. 2024/2025*
*Studente: Massimiliano Camillucci*
