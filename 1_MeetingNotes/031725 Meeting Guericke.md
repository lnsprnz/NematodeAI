## 17.03.2025 | Abgleich Guericke Prinz

Teilnehmer: LP, Stephan Guericke

# Problemstellung
Ziel: Analyse der Population Dynamics in der FuE von neuen Nematoden (3 Klassen: Larven, infektiöse Larven und Erwachsene Nematoden)

<img width="304" alt="image" src="https://github.com/user-attachments/assets/82a18e1e-0542-4626-9372-c3bcb3251195" />

- Unterscheidungen zwischen Nematoden
- morphologische merkmale --> hülle um den nematoden & größe
- bewegungsmerkmale --> "grasende merkmale"

# Vorstellung der Literaturrecherche
- In der Literatur werden verschiedenen Hardware Setups entwickelt
    - eigens Hardwaredesign (Erkennung auf Petrischalen)
- Softwaresetup
    - eigentlich fast alle Modelle, hauptsächlich Klassifikation und Objektdetektion
    - Eine Arbeit bei der zeitliche Aspekte miteinbezogen werden (Erkennung ob Bewegung oder nicht)
    - Vorverarbeitung (Cany Edge Detection)
    - CLAH (Contrat Enhancment)
    - watershed
- diverse Datensätze welche nur teilweise Schnittmengen mit der Fragstellung haben

# Vorstellung der ersten Ansätze
- Daten Aufnahme und Annotation
    - Beispieldaten zeigen
    - Termin bei E-Nema 
- Technisches Ansätze
    - YOLO
    - Mask R-CNN
    - SAM (Meta)
- Großer Aufwand wird die Annotation an der man wg. der Evaluation nicht herumkommt. Hier könnte Vorverabeitung oder SAM eine Unterstützung sein
- Drei Ansätze:
    - straightforward: 
        - Daten annotieren und YOLO bzw. Mask R-CNN trainieren ggf. Vorverarbeitung
    - Classifications: 
        - Verwendung von SAM um Daten annotation zu vereinfachen (Auto Segmentation via Koordinaten Prompt) 
        - Verwendung eine bereits trainierten Models oder SAM parameter finetunen um Bounding Boxen Nematoden (jeglicher Klasse zu erkennen)
        - Dann ein Klassifikationsmodell auf ROIs trainieren
    - Keypoint Detection & Unsupervised clustering
        - Keypoints Predicten (3 o. 5 Keypoints auf Körper der Nematoden)
        - Features: Bewegung der Keypoints über x Frames, Abstand der Keypoints, etc.
        - Ziel: Bewegungsmerkmale und Morphologische Merkmale der einzelnen Nematoden analysieren
        - iterativ beginnen mit hohen verdünnungen

# Training auf den HS Servern

