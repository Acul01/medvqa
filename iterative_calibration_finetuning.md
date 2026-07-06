# Anleitung für Coding-Agent: Iterative Calibration Fine-Tuning

## Ziel

Implementiere **eine neue, eigenständige Trainingspipeline** für
iteratives Calibration Fine-Tuning. **Bestehende Dateien dürfen nicht
verändert werden.** Die Implementierung soll ausschließlich neue Dateien
anlegen und vorhandene Komponenten des Repositories wiederverwenden.

## Neue Dateien

Mindestens:

-   `iterative_calibration_finetuning.py`
-   optional `calibration_losses.py`

## Grundprinzip

Das Verfahren orientiert sich am Warm-Start Active Learning.

-   Modell wird **nicht** neu initialisiert.
-   Nach jeder Iteration wird ein **neuer
    Calibration-Trainingsdatensatz** erzeugt.
-   Fine-Tuning erfolgt immer auf Basis der Gewichte der vorherigen
    Iteration.

## Voraussetzungen

Ein bestehender Datensatz liegt bereits vor und enthält mindestens:

-   image
-   question
-   ground_truth_answer
-   generated_answer
-   generated_rationale
-   confidence
-   entropy

## Ablauf

### Iteration 0

1.  Lade den vorbereiteten Datensatz.
2.  Teile ihn in
    -   Active-Learning-Pool
    -   Holdout/Evaluation
3.  Erzeuge einen Calibration-Trainingsdatensatz.

### Auswahlregeln

80 % der Beispiele:

-   50 %:
    -   richtige Antwort
    -   hohe Entropie (niedrige Confidence)
-   50 %:
    -   falsche Antwort
    -   niedrige Entropie (hohe Confidence)

20 %:

-   zufällige Beispiele aus dem Pool

Der Anteil X % der ausgewählten Beispiele muss über argparse
konfigurierbar sein.

Beispiel:

    --selection_ratio 0.10

bedeutet:

-   oberste 10 %
-   davon
    -   5 % correct + high entropy
    -   5 % wrong + high confidence

### Fine-Tuning

Warm Start.

Das Modell wird **nicht neu initialisiert**.

Nach jeder Iteration wird mit den aktuellen Gewichten weitertrainiert.

## Weitere Iterationen

Nach jeder Iteration:

1.  Inference auf dem aktuell definierten Active-Learning-Pool
2.  Confidence und Entropie neu berechnen
3.  Neuen Calibration-Datensatz erzeugen
4.  Weiteres Fine-Tuning

Der Calibration-Datensatz wird **nicht akkumuliert**.

Jede Iteration erzeugt einen vollständig neuen Datensatz.

## Loss

Implementiere eine modulare Struktur

    loss =
    answer_loss
    +
    lambda_cal * calibration_loss

Anforderungen:

-   Answer Loss austauschbar
-   Calibration Loss austauschbar
-   lambda über argparse

Die konkrete Calibration-Loss-Implementierung soll modular vorbereitet
werden.

## Modulstruktur

Die Datei sollte ungefähr folgende Struktur besitzen:

    parse_args()
    load_dataset()
    split_dataset()
    run_inference()
    compute_confidence_entropy()
    identify_calibration_errors()
    sample_calibration_dataset()
    build_dataloader()
    build_losses()
    train_iteration()
    evaluate_iteration()
    main()

## Wiederverwendung

Bestehende Funktionen aus dem Repository sollen importiert werden.

Keine Logik duplizieren.

## Logging

Pro Iteration speichern:

-   Accuracy
-   Calibration Loss
-   Answer Loss
-   AURC
-   ECE (falls vorhanden)
-   Anzahl ausgewählter Samples
-   Anteil correct/high entropy
-   Anteil wrong/high confidence
-   Anteil Random

## CLI

Mindestens folgende Argumente:

    --selection_ratio
    --random_ratio
    --iterations
    --lambda_cal
    --batch_size
    --learning_rate
    --output_dir

## Wichtige Randbedingungen

-   Keine bestehenden Dateien verändern.
-   Nur neue Dateien anlegen.
-   Bestehende Trainingspipeline möglichst wiederverwenden.
-   Code klar modular halten.
-   Jede Funktion dokumentieren.
-   Typannotationen verwenden.
-   Reproduzierbarkeit über festen Seed sicherstellen.
-   Keine Hardcodierung von Pfaden.
