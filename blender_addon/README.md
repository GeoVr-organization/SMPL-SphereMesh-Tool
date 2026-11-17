# Sphere Plane Fit — add-on Blender

Fitta una sfera tangente a un insieme di vertici di una mesh, con il sistema
della sezione 8 di `SMFitting/main.py` ridotto a una sola sfera con tutti i
contributi a 1.

## Installazione

Blender 4.2 o successivo: *Edit → Preferences → Get Extensions → ▾ → Install
from Disk…* e scegli `sphere_plane_fit-0.2.0.zip`. Il pannello è in
*View3D → Sidebar (N) → Sphere Fit*.

Dopo aver modificato `sphere_plane_fit/__init__.py`, rigenera lo zip:

```
cd blender_addon/sphere_plane_fit && zip -r ../sphere_plane_fit-0.2.0.zip __init__.py blender_manifest.toml
```

## Uso

1. Edit Mode sulla mesh, seleziona i vertici, **Crea sfera dai vertici
   selezionati**. Nascono un vertex group `SPF_Sphere…` sulla mesh e un
   oggetto sfera con lo stesso nome nella collection *Sphere Fit*. La sfera
   parte dal baricentro dei vertici, con raggio la loro distanza media, e
   viene subito fittata (disattivabile nel pannello *Adjust Last Operation*).
2. **Blocca X / Y / Z / R** tiene quella componente al valore attuale della
   sfera. Per fissare y a un valore preciso: sposta la sfera, poi blocca Y.
   Con *Rifitta automaticamente* il fit riparte a ogni blocco o sblocco.
   Dopo aver spostato a mano una sfera con blocchi attivi, premi **Fit**.
3. Per cambiare i vertici: seleziona la sfera, **Modifica vertici** (entra in
   Edit Mode sulla mesh con i vertici della sfera selezionati), cambia la
   selezione e usa **Sostituisci / Aggiungi / Rimuovi**; **Fine** torna alla
   sfera. In Edit Mode il campo *Sfera* sceglie quale sfera della mesh stai
   modificando.
4. **Sfera iniziale** riporta la sfera a baricentro + distanza media.
   Il cestino elimina la sfera e, se vuoi, il suo vertex group.

## Gruppi colorati

Pannello *Gruppi colorati*, sotto quello delle sfere, per la mesh attiva:
vertex group con un nome e un colore, da riselezionare quando serve.

- **+** crea un gruppo; in Edit Mode ci mette i vertici selezionati.
  **−** elimina il gruppo e il suo vertex group.
- Nella lista: clic sul colore per cambiarlo, doppio clic sul nome per
  rinominarlo (rinomina anche il vertex group). Il numero è quanti vertici ha.
- In Edit Mode: **Assegna** mette i vertici selezionati nel gruppo attivo e li
  toglie dagli altri gruppi colorati (ogni vertice ha un solo colore); **Togli**
  li toglie dal gruppo; **Seleziona / Deseleziona** i vertici del gruppo.
- I colori sono l'attributo `SPF_GroupColor` della mesh, aggiornato a ogni
  modifica. **Mostra colori** imposta il viewport in Solid con *Color:
  Attribute*; **Nascondi** torna a *Material*. I vertici senza gruppo sono grigi.
- Gli altri vertex group (quelli delle sfere, per esempio) non vengono toccati.

## Il sistema

Per ogni vertice `p_i` con normale `n_i` (coordinate mondo, mesh base senza
modificatori):

```
c · n_i + r = p_i · n_i        incognite: cx, cy, cz, r
```

risolto ai minimi quadrati: minimizza la distanza tra la sfera e il piano
tangente di ogni vertice. Non c'è regolarizzazione. Una componente bloccata
esce dal sistema: la sua colonna, moltiplicata per il valore attuale, passa a
destra (`b_i -= n_iy · y0` per y). È il vincolo esatto, il limite di `P → ∞`
in `main.py`. Togliere la colonna senza spostarla a destra vorrebbe dire
fissare y = 0.

Il sistema si risolve per lo spostamento dal valore attuale: se i vertici non
determinano una direzione (patch piatta: solo `c·n + r` è fissato), quella
direzione resta dov'è e il pannello lo segnala come *degenere*.

## Memoria

- I vertici sono il vertex group sulla mesh: lo vedi e lo modifichi anche con
  gli strumenti nativi (Object Data → Vertex Groups) e sopravvive alle
  modifiche di topologia.
- La sfera tiene un puntatore all'oggetto mesh e il **nome** del gruppo
  (Blender non permette puntatori a un vertex group). Se rinomini il gruppo,
  riscegli il nome nel campo *Gruppo* del pannello.
- Il raggio è la scala dell'oggetto sfera (mesh di raggio 1): non applicare la
  scala (Ctrl+A) alle sfere.

## Note

- Normali: quelle di Blender, pesate per angolo; `ObjMesh.py` somma le normali
  del file .obj o pesa per area. Le differenze sono piccole.
- I vertici senza facce vengono ignorati (Blender dà loro come normale la
  posizione normalizzata).
- Se il raggio risultante è ≤ 0 (patch concava o normali invertite) la sfera
  non viene spostata e il pannello mostra l'errore.

## Test

```
pip install bpy        # stessa versione di Python di bpy
python blender_addon/tests/test_headless.py
```
