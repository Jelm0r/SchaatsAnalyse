# Te doen na de crash van 13-8-2026 (knippen + batch)

1. **Kiezers opruimen.** `DoelKiezer`, `HorizonKiezer`, `KalibratieKiezer`, `FragmentKiezer`:
   `setAttribute(Qt.WA_DeleteOnClose)` of `deleteLater()` na `exec()` — o.a. in de
   verzamellus van `_nieuwe_batch_analyse` en in `_knip_opname`.
2. **Knip-voortgang fijner.** In `_knip_naar_tijdelijk` (`_melden`) niet in hele procenten
   melden maar in promille, of `setValue` op frame-basis met een tijdsdrempel.
3. **Crashlog.** In `main()` van `schaats_gui.py`: `faulthandler.enable(<logbestand>)` +
   een `sys.excepthook` die naar datzelfde bestand schrijft.
4. **Wisselbestand.** Op deze laptop handmatig op ~16 GB zetten (staat nu op 2 GB).
