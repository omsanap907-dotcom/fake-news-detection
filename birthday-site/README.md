# Krati Birthday Universe

A mobile-first, non-romantic birthday microsite concept for Krati, focused on astronomy and a birthday time capsule.

## Experience
- Cosmic opening screen with a launch button
- Interactive starfield and custom artistic constellation (clearly marked as fictional art)
- Birthday sky information: Moon phase, planetary positions, sunrise/sunset, and constellations when supplied with a date, time, and location
- Earth journey counter and age countdown
- This-day-in-history section
- Birthday cake/confetti celebration

## Accuracy rules
- Do not invent the birth year, time, birthplace, planetary alignments, historical events, or astronomy values.
- Make the birth date, time, and location configurable.
- If exact birth time/location are not provided, show which facts can be calculated and label estimates/limitations clearly.
- Cite data providers inline and include a sources panel for astronomy/history facts.
- The custom "Krati constellation" is an artistic design, not an officially recognized constellation.

## Suggested implementation
Use plain HTML/CSS/JavaScript for easy hosting on Netlify. A lightweight canvas starfield can avoid heavy libraries. If accurate astronomical calculations are needed, use a maintained astronomy library or reputable API and document its source. Keep the page mobile-first, accessible, performant, and usable without autoplay audio.

## Current status
This is a project brief only. The parent repository is a fake-news detection learning project; this folder does not modify or integrate with its Python model. A dedicated repository would be cleaner for the actual website implementation.
