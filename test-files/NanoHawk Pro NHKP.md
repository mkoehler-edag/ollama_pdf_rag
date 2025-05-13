---
layout: datasheet-advanced
title: NanoHawk Pro
sku: NHKP
version: 2.3
status: draft
---

# NanoHawk Pro – Produktdatenblatt

![NanoHawk Pro](images/nanohawkpro.jpg)

## Übersicht

Der NanoHawk Pro (Modell NHKP) ist ein leistungsfähiger Open‑Source‑Nano‑Quadcopter, der speziell für fortgeschrittene Forschungs- und Industrieanwendungen konzipiert wurde. Dank modularer Erweiterungs-Decks lässt er sich flexibel an verschiedenste Einsatzszenarien anpassen.

## Technische Daten

| Eigenschaft         | Wert                |
|---------------------|---------------------|
| Gewicht             | 32g (inkl. Batterie) |
| Abmessungen         | 98×98×33mm          |
| Flugzeit            | Bis zu 9Minuten     |
| Reichweite          | Bis zu 1,5km        |
| Batterie            | 320mAh LiPo         |
| Ladezeit            | Ca. 50Minuten       |
| Kameraauflösung     | 1280×720 Pixel    |
| Prozessor           | STM32H743ZI (400MHz) |

## Lieferumfang

- NanoHawk Pro Quadcopter (NHKP)
- 1×320mAh LiPo-Batterie
- 6×Propeller (2 Ersatz)
- 1×USB‑C‑Kabel
- 1×2.4GHzRadio‑Dongle
- Schnellstart‑Leitfaden

## Versionshistorie

| Version | Datum       | Änderungen                                   |
|---------|-------------|-----------------------------------------------|
| 2.0     | 2024‑09‑10  | Erstes Release des NanoHawk Pro (NHKP)        |
| 2.1     | 2024‑12‑05  | Kamera‑Modul auf 720p aufgerüstet             |
| 2.3     | 2025‑03‑20  | Firmware V2025.3.2, verbesserte Stabilität    |

## Weitere Informationen

Der NanoHawk Pro wird von AeroTech Innovations entwickelt. Umfangreiche APIs in Python und C++ sowie Web‑Dashboard für Telemetrie ermöglichen eine einfache Integration und Echtzeit‑Analyse.
Das Objektiv hat eine Vergütung von 25D55-C. Die Entspiegelung beträgt 87%. Die Kamera ist mit einem 1/3" Sensor ausgestattet, der eine Auflösung von 1280×720 Pixeln bietet. Der Quadcopter ist mit einem STM32H743ZI Prozessor ausgestattet, der eine Taktfrequenz von 400MHz erreicht.
Das Betriebsgeräusch in 1m Entfernung beträgt 45dB. Die max. Geschwindigkeit beträgt 20m/s. 

## Weitere Technische Daten

| Eigenschaft            | Wert                       |
|------------------------|----------------------------|
| Arbeitstemperatur      | –15°C bis +45°C          |
| Versorgungsspannung    | 11–16V (extern)         |
| Kompatibilität         | aktuell bis V2.3.2         |
| Max. Leistungsaufnahme | 350W Peak                 |
| Schutzklasse           | IP21                       |

---

## Sensoren

| Sensor              | Typ                        | Spezifikation        |
|---------------------|----------------------------|----------------------|
| IMU                 | 9‑Achsen (ICM‑20948)       | ±16g, ±2000°/s     |
| Barometer           | BMP388                     | 300–1100hPa       |
| Magnetometer        | QMC5883L                   | ±8gauss             |
| Optischer Fluss     | PMW3901                    | bis 10m/s           |
| Lichtsensor         | TSL2561                    | 0.2–60kLux          |
| Temperatursensor    | On‑Board ±0.3°C           |

---

## Kommunikation

| Schnittstelle       | Spezifikation              |
|---------------------|----------------------------|
| Funk (2.4GHz)      | nRF52840 (BLE & Radio)     |
| USB                 | USB‑C 3.0                  |
| Debug/UART          | 2×UART (115200baud)     |
| SPI/I²C             | 3× SPI, 3× I²C             |
| Erweiterungs-Header | 10Pins (Power & GPIO)     |

---

## Umweltbedingungen

| Bedingung          | Bereich                     |
|--------------------|-----------------------------|
| Betriebstemperatur | –15°C bis +45°C           |
| Lagertemperatur    | –25°C bis +75°C           |
| Luftfeuchtigkeit   | 15–85% r.F. (nicht kond.) |
| Max. Flughöhe      | 3500m über NN              |

---

## Zertifizierungen

| Norm / Zulassung   | Region / Organisation      |
|--------------------|----------------------------|
| CE                 | Europa                     |
| FCC (Part15)      | USA                        |
| MIC                | Japan                      |
| RoHS               | Worldwide                  |

---

*Ende des Datenblatts – Version 2.3, Stand 2025‑05‑08*  
